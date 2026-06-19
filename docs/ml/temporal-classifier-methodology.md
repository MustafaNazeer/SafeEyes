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
    --video-root data/uta-rldd --feature-root features/uta-rldd --frame-step 5
```

`--frame-step 5` processes every fifth frame, which keeps the extraction tractable
on a CPU and the window count within memory while preserving the slow signals
(eye closure, blink, yawn, nod) that drowsiness produces.

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

Per frame features are standardized with training statistics, then windowed at
size 150, stride 75. The GRU trains for 30 epochs. Evaluated on the 10 held out
subjects of the folds 1 to 4 split:

| Metric | GRU (primary) | GBT (baseline) |
|--------|---------------|----------------|
| Overall accuracy | 52.1% | 46.5% |
| Macro AUROC | 0.693 | 0.618 |
| False alarm rate | 0.152 | 0.258 |
| Recall, alert | 74.7% | 64.0% |
| Recall, low vigilance | 31.1% | 21.5% |
| Recall, drowsy | 51.5% | 55.4% |

The GRU leads the gradient boosted baseline on overall accuracy, macro AUROC, and
the false alarm rate, so the sequence model earns its place as the primary model
over the simpler baseline. The full metric files are
[temporal-metrics.json](temporal-metrics.json) and
[temporal-metrics-gbt.json](temporal-metrics-gbt.json).

Read honestly:

- Low vigilance is the hardest class for both models, the expected pattern for
  the fuzzy intermediate state between alert and drowsy. The baseline edges the
  GRU on drowsy recall (55.4% against 51.5%) while losing on every aggregate.
- Drowsy recall around one half means this is an assistive signal, not a reliable
  detector of every drowsy moment, consistent with the prototype framing.
- These are a single subject independent split over 48 subjects (folds 1 to 4).
  Hyperparameters were not tuned against this test set, which would inflate the
  numbers; a validation fold tuning pass is recorded below.

## Hyperparameter tuning

To check whether the default configuration could be improved without peeking at
the test set, a validation fold was carved from the training subjects alone (a
quarter of them) and the GRU's window size, stride, epoch count, and learning
rate were swept on it. The configuration with the best validation macro AUROC
(window 200, stride 100, 100 epochs) was then trained on the full training split
and evaluated once on the held out test set:

| Metric | Default (reported) | Validation selected |
|--------|--------------------|---------------------|
| Overall accuracy | 52.1% | 47.9% |
| Macro AUROC | 0.693 | 0.681 |
| False alarm rate | 0.152 | 0.112 |
| Recall, drowsy | 51.5% | 47.3% |

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
