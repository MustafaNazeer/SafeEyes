# Temporal fatigue classifier methodology

How the second stage turns a window of per frame signals into a fatigue level,
and how that level is evaluated honestly. For the system overview see
[docs/architecture.md](../architecture.md); for the dataset and split details see
[docs/source/datasets.md](../source/datasets.md); for the split correctness sign
off see [docs/ml/validation-notes.md](validation-notes.md).

## What the stage does

The perception stage produces a per frame feature vector (eye aspect ratio,
mouth aspect ratio, head pose, and related signals). These accumulate in a fixed
length rolling window. The window is then classified into one of the three
UTA-RLDD fatigue levels: alert, low vigilance, drowsy.

## Features

Two views of a window are used:

- The raw per frame feature sequence, fed directly to the sequence model.
- Aggregated window features for the baseline: the proportion of eye closure
  over the window, blink and yawn and head nod event counts and per minute
  rates, mean blink duration, and mean signal levels. An event is counted once
  per onset (a transition into the state), so a sustained closure is one event,
  not one per frame. These definitions are unit tested against known sequences.

## Models

- **Primary: a small recurrent network (GRU).** It reads the per frame feature
  sequence and outputs a fatigue level. Kept compact so it can run in real time
  on the edge device after quantization.
- **Baseline: gradient boosted trees.** A simple, honest point of comparison
  trained on the window features. Reporting the primary model next to a plain
  baseline guards against overstating how much the sequence model actually buys.

## Evaluation

All numbers are computed on a subject independent UTA-RLDD split: no subject
appears in both training and evaluation, so the metrics reflect performance on
drivers the model never saw. The reported metrics are:

- **Per class accuracy** (recall per fatigue level), so a model that does well on
  the easy class but misses the dangerous one cannot hide behind an average.
- **Macro AUROC** across the three classes (one versus rest), giving a threshold
  independent view of separability.
- **False alarm rate**, reported beside accuracy and never hidden. It is defined
  as the fraction of genuinely not drowsy windows (alert or low vigilance) that
  are classified as drowsy. A detector that cries wolf is one a real driver would
  switch off, so this number is treated as first class.

## Reproducibility

The full pipeline is three reproducible steps on the fixed split.

Build the subject independent split manifests from the downloaded videos:

```
python -m safeeyes.data.build_splits \
    --dataset uta-rldd --root data/uta-rldd --out splits/uta-rldd --ratios 0.8 0.0 0.2
```

The split is 80/20 at the subject level (no validation fold, since training uses
train and test only), seed 0. With the folds 1 to 4 coverage of the Kaggle mirror
(48 subjects, see [docs/source/datasets.md](../source/datasets.md)) this is 38
training and 10 held out test subjects.

Extract the per frame feature sequence for every clip, driving the same
perception path the live runtime uses (FaceMesh landmarks then the fixed feature
vector). Frames with no detected face are skipped, exactly as the runtime skips
them, so the training features match what the pipeline sees live. The step is
resumable, so an interrupted run continues where it stopped:

```
python -m safeeyes.perception.extract \
    --manifest splits/uta-rldd/train.csv splits/uta-rldd/test.csv \
    --video-root data/uta-rldd --feature-root features/uta-rldd --frame-step 3
```

`--frame-step 3` processes every third frame, about ten features per second. That
matches the rate at which the live runtime fills its window buffer, so a 150 frame
window spans the same amount of time in training as it does live, roughly fifteen
seconds. An earlier version sampled every fifth frame (about six features per
second), which made a 150 frame window cover about twenty five seconds in training
while the live loop filled the same window in about half that. Extracting at the
live cadence removes that train and serve mismatch, at the cost of more frames and
a longer extraction.

Train and evaluate the classifier on the extracted features:

```
python -m safeeyes.temporal.train_temporal \
    --train-manifest splits/uta-rldd/train.csv \
    --val-manifest splits/uta-rldd/test.csv \
    --feature-root features/uta-rldd \
    --model gru --metrics-out docs/ml/temporal-metrics.json
```

The harness windows the saved feature arrays, trains, and writes the metrics
file. Every figure below traces back to that command on the fixed split and the
recorded feature extraction rather than to a remembered value.

## Results

Per frame features are standardized with training statistics, which are baked
into the deployed checkpoint, then windowed at size 150, stride 75. The GRU trains
for 30 epochs. Evaluated on the 10 held out subjects of the folds 1 to 4 split,
the deployed checkpoint (seed 0):

| Metric | GRU (primary) | GBT (baseline) |
|--------|---------------|----------------|
| Overall accuracy | 47.1% | 46.3% |
| Macro AUROC | 0.653 | 0.641 |
| False alarm rate | 0.100 | 0.285 |
| Recall, alert | 68.5% | 63.1% |
| Recall, low vigilance | 24.9% | 20.4% |
| Recall, drowsy | 48.9% | 56.8% |

The full metric files are [temporal-metrics.json](temporal-metrics.json) and
[temporal-metrics-gbt.json](temporal-metrics-gbt.json).

Recurrent training is sensitive to initialization, so the GRU was trained across
five seeds on the same split to size that variance honestly. The deployed seed 0
run sits at the favorable end of the range:

| Seed | Accuracy | Macro AUROC | False alarm rate | Drowsy recall |
|------|----------|-------------|------------------|---------------|
| 0 (deployed) | 47.1% | 0.653 | 0.100 | 48.9% |
| 1 | 44.0% | 0.613 | 0.181 | 49.9% |
| 2 | 45.4% | 0.616 | 0.121 | 47.9% |
| 3 | 45.0% | 0.629 | 0.119 | 48.3% |
| 4 | 41.3% | 0.610 | 0.211 | 47.3% |
| mean | 44.6% | 0.624 | 0.147 | 48.5% |
| std | 1.9% | 0.016 | 0.042 | 0.9% |

The honest headline is the mean and spread, not the single deployed run: about
44.6% accuracy, macro AUROC about 0.624, drowsy recall about 48.5%, and a false
alarm rate that ranges from 0.100 to 0.211 across seeds, mean 0.147.

Read honestly:

- The GRU does not robustly beat the gradient boosted baseline. The deployed seed
  0 run leads it on every aggregate, but averaged over seeds the GRU trails the
  baseline on accuracy (44.6% against 46.3%) and macro AUROC (0.624 against
  0.641). Where the GRU wins consistently is the false alarm rate (mean 0.147
  against 0.285), less than half the baseline's. It is retained as the deployed
  model on that basis, a first class metric for a drowsiness alerter, and not on
  an accuracy advantage it does not reliably hold.
- Earlier versions of this project reported a larger GRU lead (54.5% accuracy and
  macro AUROC 0.706 at a false alarm rate of 0.139). That gap did not survive two
  corrections made since. Features are now extracted at the live cadence (above),
  and a head pose feature that had been dominated by an angle wraparound artifact
  rather than real head motion was fixed. Trained on honest features at the rate
  the model actually runs, the sequence model's advantage over the baseline is
  mostly a lower false alarm rate, and the earlier headline is superseded by the
  numbers here.
- Low vigilance is the hardest class for both models, the expected pattern for
  the fuzzy intermediate state between alert and drowsy.
- Drowsy recall around one half means this is an assistive signal, not a reliable
  detector of every drowsy moment, consistent with the prototype framing.
- These are a single subject independent split over 48 subjects (folds 1 to 4).
  Hyperparameters were not tuned against this test set, which would inflate the
  numbers; a validation fold tuning pass is recorded below.

## Cross-dataset evaluation on DMD drowsiness

The deployed model was evaluated, with no retraining, on the DMD drowsiness bundle
(16 face-camera recordings from 13 subjects, session s5). None of these subjects
or frames were seen in training, so this is a genuine cross-dataset generalization
test. DMD ships no alert, low vigilance, or drowsy labels, so the model's three
class output is collapsed to drowsy versus not, and the ground truth is a frame
level drowsy label derived from the annotations: a frame is drowsy when it lies
inside a sustained eye closure (an eyes closed interval lasting at least 0.5
seconds, a microsleep proxy), with ordinary blinks and yawning excluded. Features
are extracted with the same perception pipeline, at frame step 3 so the sampling
rate matches the deployed model, and windowed at size 150, stride 75; a window
counts as drowsy when at least a set fraction of its frames carry the drowsy label.
Matching the extraction cadence keeps the cross-dataset test consistent with how
the model runs at serving time.

Two limitations bound every number here. The DMD drowsiness was acted, not genuine
fatigue, so this does not measure detection of real drowsiness. And the drowsy
label is a sustained eye closure proxy from the annotations, not an independent
drowsiness ground truth.

At the primary 10 percent window threshold (367 windows, 146 of them drowsy), the
model reaches 91.8% drowsy recall but a false alarm rate of 0.548, for 63.8%
overall accuracy. The full metric file is
[temporal-cross-dmd-metrics.json](temporal-cross-dmd-metrics.json). Because the
window threshold changes how many windows count as drowsy, the table sweeps it:

| Window drowsy threshold | Drowsy windows | Recall | False alarm rate | Accuracy |
|-------------------------|----------------|--------|------------------|----------|
| 5 percent | 190 | 0.879 | 0.497 | 0.698 |
| 10 percent | 146 | 0.918 | 0.548 | 0.638 |
| 20 percent | 51 | 0.961 | 0.652 | 0.433 |
| 50 percent | 0 | n/a | 0.695 | 0.305 |

Read honestly: the model still over triggers on DMD, firing drowsy on roughly
seven in ten windows, so recall stays high while the false alarm rate climbs as the
true negative pool grows. It does not generalize cleanly to this different
distribution of acted closures, camera, and lighting. It is, however, less extreme
than before the model was corrected. An earlier version of this evaluation, run on
the model before it was retrained at the live cadence and before the head pose
wraparound was fixed, gave 97.7% recall at a false alarm rate of 0.634 for 62.1%
accuracy; the corrections lowered the false alarm rate to 0.548 and raised accuracy
to 63.8%. That matches the finding that the earlier extreme over-firing was
substantially an artifact of a train and serve mismatch, while genuine transfer to
DMD stays weak. The correction narrows the gap, it does not close it. This
reinforces the safety review's standing concern about the false alarm rate, and on
this evidence the model is not ready to be trusted on DMD-like data, so no claim of
dependable cross-dataset detection is made.

## DMD-inclusive training (recorded, not adopted)

Since the model over-fires on DMD, a natural question is whether adding DMD
drowsiness data to the training set improves generalization. This was tested,
measured, and then not adopted.

DMD frame-level drowsy labels were mapped onto the three UTA classes: a window is
drowsy when at least ten percent of its frames fall in a sustained eye closure,
otherwise alert, since DMD has no low vigilance state. The 13 DMD subjects were
split subject independently into 9 for training and 4 held out. A model trained on
UTA plus the 9 DMD subjects was compared against the UTA-only model, both evaluated
on the UTA test set and on the 4 held-out DMD subjects, across five seeds.

| Metric | UTA-only | UTA plus DMD |
|--------|----------|--------------|
| UTA test accuracy (seed mean) | 44.5% | 44.1% |
| UTA test macro AUROC (seed mean) | 0.624 | 0.623 |
| Held-out DMD false alarm rate (seed mean) | 0.662 | 0.600 |

Adding DMD left the UTA numbers unchanged within seed noise and lowered the
held-out DMD false alarm rate on average, from 0.662 to 0.600. But the gain was
small and inconsistent: across five seeds one improved sharply, three improved
modestly, and one regressed, and the model still over-fired on DMD, with a false
alarm rate near 0.6. Because a deployed system ships one checkpoint rather than the
seed average, and the checkpoint chosen by the fixed seed convention showed almost
no improvement, the trade was not worth giving up DMD as a fully independent
cross-dataset test. The UTA-only model is retained, and this experiment is recorded
rather than discarded, consistent with the hyperparameter tuning result below.

## Hyperparameter tuning

To check whether the default configuration could be improved without peeking at
the test set, a validation fold was carved from the training subjects alone (a
quarter of them) and the GRU's window size, stride, epoch count, and learning
rate were swept on it. The configuration with the best validation macro AUROC
(window 200, stride 100, 100 epochs) was then trained on the full training split
and evaluated once on the held out test set:

| Metric | Default (at the time of the sweep) | Validation selected |
|--------|------------------------------------|---------------------|
| Overall accuracy | 52.1% | 47.9% |
| Macro AUROC | 0.693 | 0.681 |
| False alarm rate | 0.152 | 0.112 |
| Recall, drowsy | 51.5% | 47.3% |

This comparison is a historical record of the sweep: both columns were measured
against the checkpoint current when it ran, whose feature extraction and head pose
feature both predate the corrections described in Results. The window, stride, and
epoch settings it retained are those of the deployed model, but the deployed
numbers in Results are lower after those corrections and are the ones to read; the
sweep is kept only as the record of a negative tuning result.

The validation selected configuration did not beat the default on accuracy or
macro AUROC. It lowered the false alarm rate, but at the cost of lower drowsy
recall, the wrong trade for a drowsiness detector. The validation differences
between configurations were also small relative to the size of the 48 subject
dataset, so the selection was weak. The default configuration is therefore
retained as the reported model. This negative result is recorded rather than
discarded, and no configuration was ever selected by looking at the test set.

## Limitations

- The UTA-RLDD labels are self reported predominant states over a clip, so a
  window label is the clip level label; brief within clip state changes are not
  separately annotated.
- The results are on folds 1 to 4 of UTA-RLDD (48 subjects), the coverage of the
  Kaggle mirror used, not the full 60 subjects, and on a single subject independent
  split rather than full cross validation. The held out set is 10 subjects, so the
  estimate is honest but has the variance of a small test set; this is stated rather
  than hidden.
- Real time behaviour, alert escalation, and the cost of a missed detection are
  the concern of the alert stage and are documented with that stage, not here.
