# Model card: yawn event classifier (not deployed)

The headline finding is about the geometric rule, not the network. Adding a
minimum duration requirement to the existing mouth aspect ratio rule turned a
signal that could not tell yawning from talking into one that mostly can, and a
trained mouth crop classifier, evaluated against that rule on the same held out
subjects by the same scoring code, did not beat it. The classifier is therefore
recorded here as a measured negative result and is **not deployed**.

For the system overview see [docs/architecture.md](../architecture.md); for the
dataset, license, and crop derivation see
[docs/source/datasets.md](../source/datasets.md); for the decision not to ship
the classifier see
[ADR 0006](../adr/0006-geometric-duration-rule-over-mouth-crop-cnn.md). Every
number below traces to [yawn-model-metrics.json](yawn-model-metrics.json) in
this directory.

## What this is and what it is not

The classifier is a frozen `mobilenet_v3_small` ImageNet backbone with a small
trainable head. Candidate mouth opening events are proposed by the geometric
rule; five mouth crops sampled across an event are pushed through the backbone
in eval mode, the per crop features are mean and max pooled into one vector per
event, and only the head is trained on those cached vectors. A video is
predicted yawning if any of its proposed events scores at or above the
checkpoint's decision threshold.

It is not a fatigue detector and not a driver state estimator. A yawn is one
weak cue among several. This is part of an assistive prototype: it makes no
diagnostic or safety of life claim, does not decide whether anyone is fit to
drive, and holds no safety certification. Precision, recall, and the talking
false positive rate are reported together everywhere below; a precision figure
never appears without its recall beside it.

## Dataset and citation

YawDD, Mirror camera set, used under its research license, which requires the
citation:

S. Abtahi, M. Omidyeganeh, S. Shirmohammadi, and B. Hariri, "YawDD: A Yawning
Detection Dataset", Proc. ACM Multimedia Systems, Singapore, March 19 to 21,
2014, pp. 337 to 342.

## Split

The Mirror set holds 90 subjects with 2 to 7 videos each. The split is subject
independent at seed 0, ratios fixed at 0.78 and 0.22 so the population lands on
exactly 70 train and 20 test subjects. The test side is 75 videos across those
20 subjects, listed in `splits/yawdd/mirror-test.csv`. The validation fold is
carved from the 70 train subjects at the same seed rather than written to disk,
so the 20 test subjects were never read during training or threshold selection.

## Fixed parameters and their provenance

- **Detection threshold 0.616703.** The 99th percentile of all per frame mouth
  aspect ratio values across the UTA-RLDD train subjects, fixed and committed
  before any YawDD data was listed, extracted, or scored. It was not tuned on
  YawDD at all.
- **Minimum duration 14 steps.** At a frame step of 3 on 30 fps video this is
  roughly 1.4 seconds of sustained opening. It was swept on the train subjects
  only, under a selection rule fixed in advance (among durations meeting the
  recall floor, take the highest precision, ties to the smaller duration).
- **Decision threshold 0.26.** Selected on the carved validation fold and frozen
  into the checkpoint before the test subjects were scored.

## The preregistered deploy rule, stated before the results

The bar was written down as numeric constants in the evaluation code before the
test subjects were scored. Both parts had to hold for the classifier to ship:

1. **Absolute bar.** Video level precision at least 0.70 and recall at least
   0.90.
2. **Head to head bar.** Precision strictly greater than the duration
   baseline's precision, while giving up no more than 0.05 recall against it.

The rule was not restated, relaxed, or reinterpreted after the numbers were
known. A run that moves those constants is not the run that was preregistered.

## Results

All three detectors were scored on the same 75 test videos from the same 20
held out subjects by the same function, so the rows are like for like.

| Detector | Precision | Recall | Talking false positive rate |
|----------|-----------|--------|-----------------------------|
| Mouth aspect ratio rule | 0.5283 | 1.0000 | 0.7273 |
| Mouth aspect ratio rule plus minimum duration | 0.8966 | 0.9286 | 0.1364 |
| Mouth crop classifier | 0.8750 | 1.0000 | 0.1818 |

Counts behind the rows: the plain rule scores 28 true positives, 25 false
positives, 0 false negatives; the duration rule 26, 3, and 2; the classifier
28, 4, and 0.

**On the earlier published figure.** The geometric rule was previously reported
at 46.7% precision and 99.1% recall in
[alert-validation.md](alert-validation.md). That measurement covered the full
320 video Mirror population, which is a different population from these 20 test
subjects, so the first row here differs. It is a population difference, not a
disagreement between two measurements of the same thing, and the baselines were
deliberately recomputed on the test subjects so the comparison is like for like.

## Decision: blocked by the baseline, not deployed

The recorded decision is `blocked_baseline`. The classifier cleared the absolute
bar comfortably (0.8750 precision at 1.0000 recall against a 0.70 and 0.90
requirement) and then failed the head to head condition: 0.8750 does not exceed
the duration baseline's 0.8966.

What the classifier actually bought is narrow and worth stating plainly. It
recovers the two yawns the duration rule misses, taking recall from 0.9286 to
1.0000, at the cost of one additional false positive, taking precision from
0.8966 to 0.8750 and the talking false positive rate from 0.1364 to 0.1818. On
a set this small that is a difference of three videos in total. The
preregistered rule does not award that trade, and it was written before anyone
knew which way it would fall, so the classifier does not ship. The duration
requirement, which is three lines of arithmetic and costs nothing at inference
time, is the better result here.

## Label leakage found during the build, and its direction

Label leakage was found in the event proposal path while the evaluation was
being built. The feature builder had been proposing events with a function that
reads a video's label to decide which openings count, which a detector at test
time cannot do: in a video labeled as yawning it kept only the single longest
opening and dropped the rest. That rule is correct for assembling training
supervision and wrong for scoring, because a live camera offers no hint about
which mouth opening is the yawn.

A leak free proposal path, which proposes every detected run in every video and
never consults the label, was written and committed **before** the evaluation
ran. Every number in the table above comes from that clean path.

The direction of the leak matters and was derived rather than assumed. Against
the leaky path, the leak free path yields true positives greater than or equal
to the leaky path's, with false positives identical, because dropping all but
the longest opening in a yawning video can only remove candidate detections in
videos whose truth is positive and cannot change anything in a negative video.
Both recall and precision therefore rise weakly under the clean path. The
classifier's 0.8750 precision at 1.0000 recall is the weakly optimistic figure
on both axes, and it still loses to the duration baseline's 0.8966. The leak did
not cost the classifier the result.

One real consequence remains. The decision threshold of 0.26 was selected on the
validation fold through the leaky path, so the validation figures (0.85
precision at 1.00 recall, on only 17 positive videos) are **not** strictly
comparable to the test figures and are quoted here only to describe how the
threshold was chosen, never as a result.

## Caveats

- **The yawns are acted.** YawDD subjects yawn on request. These are performed
  yawns, not naturally occurring fatigue, and every number inherits that.
- **Ground truth is video level only.** The dataset ships no per frame
  annotation. Training supervision was therefore weak, and the rule that only
  the longest opening in a yawning clip counts as a positive is an assumption
  made to build a training set, not a fact recovered from the data. A different
  assumption would produce different training labels.
- **The test set is 20 subjects and 75 videos.** That is small. A difference of
  three videos separates the classifier from the baseline, and no reading of
  these numbers should be more confident than that sample supports.
- **The detector inherits the proposal stage's recall ceiling.** Candidate
  events are generated by the same preregistered threshold, so a yawn that never
  crosses it can never be recovered downstream. On the full Mirror population
  that stage measured 99.1% recall; on these test subjects it proposed an event
  in every yawning video, so the ceiling did not bind here, but it exists.
- **The training crops are not class balanced like the videos.** The extraction
  gate keeps pixels only around frames whose mouth aspect ratio clears a
  permissive pre filter. In practice it retained roughly 45 percent of a talking
  clip's rows but only 1 of 212 rows in a normal driving clip, so resting mouths
  are nearly absent from the crop set. The head was trained on a population of
  open mouths, which is not the population a live camera sees.
- **Precision is never reported alone.** Precision, recall, and the talking
  false positive rate appear together in every row above, and a regression guard
  enforces that any document reporting one of these precision figures reports its
  recall on the same line.

## Limitations and disclaimer

This is an assistive prototype, not a medical or safety of life device. It makes
no diagnostic claim and does not decide fitness to drive. Every figure here
traces to [yawn-model-metrics.json](yawn-model-metrics.json), produced by a
single frozen evaluation run on a fixed, subject independent split, and is
pinned by a regression test so it cannot drift from the committed metrics file.
