# ADR 0006: Geometric duration rule over the mouth crop classifier

- Status: Accepted
- Date: 2026-07-20

## Context

The mouth aspect ratio yawn signal was a threshold crossing rule: a video
registers a yawn if its mouth aspect ratio reaches the preregistered detection
threshold of 0.616703 at least once. Measured over the full 320 video YawDD
Mirror population it scored 46.7% precision at 99.1% recall, with 80.0% of pure
talking videos firing. Talking was the designed hard negative and it defeated
the rule, which is recorded in
[alert-validation.md](../ml/alert-validation.md). The follow up question was
whether a trained classifier over mouth pixels could separate yawning from
talking where the geometry could not.

Two candidates were prepared for a single frozen comparison on a subject
independent split of the Mirror set (70 train and 20 test subjects at seed 0,
75 test videos):

- Requiring the opening to last a minimum duration on top of the same
  threshold, with the duration swept on train subjects only under a selection
  rule fixed in advance, which chose 14 steps (roughly 1.4 seconds).
- A frozen `mobilenet_v3_small` backbone with a small trainable head over five
  mouth crops per proposed event, with its decision threshold selected on a
  validation fold carved from the train subjects and frozen at 0.26 before the
  test subjects were read.

The deploy bar was written down as numeric constants before the test subjects
were scored: precision at least 0.70 and recall at least 0.90, and precision
strictly greater than the duration baseline's while giving up no more than 0.05
recall against it.

Label leakage was found in the event proposal path during the build. The
proposal function had been reading a video's label to decide which openings
count, which a detector at test time cannot do. A leak free proposal path was
written and committed before the evaluation ran, so every number below comes
from clean code.

## Decision

**The classifier is not deployed. The geometric rule with a minimum duration is
the yawn signal.**

All three detectors were scored on the same 75 held out videos by the same
function. From [yawn-model-metrics.json](../ml/yawn-model-metrics.json):

| Detector | Precision | Recall | Talking false positive rate |
|----------|-----------|--------|-----------------------------|
| Mouth aspect ratio rule | 0.5283 | 1.0000 | 0.7273 |
| Mouth aspect ratio rule plus minimum duration | 0.8966 | 0.9286 | 0.1364 |
| Mouth crop classifier | 0.8750 | 1.0000 | 0.1818 |

The recorded decision is `blocked_baseline`. The classifier cleared the absolute
bar and then failed the head to head condition, because its precision does not
exceed the duration baseline's. It recovers the two yawns the duration rule
misses at the cost of one additional false positive, a difference of three
videos in total, and the preregistered rule does not award that trade.

The duration requirement is the finding worth keeping. Requiring an opening to
persist for roughly 1.4 seconds is a few lines of arithmetic, costs nothing at
inference time, needs no model file, no backbone, and no training data, and it
moved the signal from unusable against talking to broadly serviceable against
it. A trained model that ties that on precision while winning on recall by two
videos has not earned the complexity it carries.

The direction of the leak was derived rather than assumed, and it does not
rescue the classifier. The leak free path yields true positives greater than or
equal to the leaky path's with false positives identical, so both recall and
precision rise weakly under the clean path. The classifier's figure is therefore
the weakly optimistic one on both axes and still loses.

## Consequences

- The accepted rule for the yawn signal is its existing geometric form plus the
  minimum duration requirement. No model file, no crop extraction, and no
  backbone inference enter the live loop for this signal, so its edge cost
  stays at zero beyond the geometry already computed for every frame.
- Wiring that requirement into the runtime is follow on work and has not been
  done. The live temporal feature in `safeeyes/temporal/features.py` still
  counts a yawn on the bare threshold crossing, with no duration requirement,
  and the duration aware predicate this comparison was measured with has no
  caller on any runtime path. Integration was out of scope for the evaluation
  recorded here. Until it lands, the runtime signal is the bare rule scored in
  the first row of the table above, not the second.
- The negative result is published rather than discarded, in
  [yawn-model-card.md](../ml/yawn-model-card.md), with the deploy rule stated
  before the results, the leak documented, and the caveats attached. A trained
  model losing to a simple rule is a result, not a failure to report.
- The comparison is small. Three videos separate the two detectors on a 20
  subject test set of acted yawns with video level ground truth only. This
  record fixes a decision on the evidence available; a larger or more natural
  evaluation could reasonably reverse it, and reversing it means a new record,
  not an edit to this one.
- The decision threshold of 0.26 was selected through the leaky path, so the
  validation figures behind it are not strictly comparable to the test figures.
  Any future attempt to revive the classifier reselects that threshold through
  the leak free path first.
- The signal remains one input among several to the temporal classifier and is
  not promoted to a standalone yawn detector by this record. SafeEyes stays an
  assistive prototype and makes no diagnostic or safety of life claim.
