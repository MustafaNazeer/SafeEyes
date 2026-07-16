# Model card: driver distraction classifier

An image classifier that labels a single cabin frame with the driver activity it
shows, for example safe driving, texting, reaching to the side, or talking to a
passenger. It is a secondary perception track that runs on a periodic schedule
alongside the fatigue pipeline. For the system overview see
[docs/architecture.md](../architecture.md); for the dataset, split, and license
details see [docs/source/datasets.md](../source/datasets.md); for the on device
latency measurements see [docs/perf/edge-benchmark.md](../perf/edge-benchmark.md).

## Intended use

Part of an assistive driver monitoring prototype. It estimates a coarse driver
activity label from a wide cabin view to surface distraction cues. It is not a
medical or safety of life device, makes no diagnostic claim, and does not decide
whether a driver is fit to drive. Every number below is reported next to the
imbalance and coverage caveats that qualify it, because on this split overall
accuracy is inflated by a few majority classes and must never stand alone.

## Task and taxonomy

The task is single frame classification over a pinned set of thirteen driver
action classes, taken exactly from the annotation vocabulary of the source
dataset:

`change_gear`, `drinking`, `hair_and_makeup`, `phonecall_left`,
`phonecall_right`, `radio`, `reach_backseat`, `reach_side`, `safe_drive`,
`standstill_or_waiting`, `talking_to_passenger`, `texting_left`, `texting_right`.

The taxonomy is fixed: an annotation class outside this set fails the data build
loudly rather than passing silently, so a future data revision cannot introduce
new classes unnoticed.

## Evaluation split and its coverage limits, stated first

The dataset holds 14 subjects. The split is subject independent at seed 0 with a
0.7 train, 0.0 validation, 0.3 test ratio, giving 10 train and 4 test subjects
with no subject overlap: 1,500 train and 496 test interval samples. Because the
population is small and the split is drawn at the subject level, two things must
be read before any accuracy figure:

- **The test split covers only 11 of the 13 classes.** The two rarest classes,
  `change_gear` and `standstill_or_waiting`, appear in only a few sessions each
  and landed entirely in the training subjects, so they have zero test support
  and are untestable here. No claim is made about them.
- **The class distribution is heavily imbalanced.** Three classes, `safe_drive`
  (192 test intervals), `reach_side` (128), and `talking_to_passenger` (81),
  make up 81% of the 496 test intervals. A model that predicted `safe_drive`
  for every frame would already score 38.71% overall accuracy while learning
  nothing. For that reason balanced accuracy (the mean of the per class recalls)
  and the per class recall table are the honest headline, and overall accuracy
  is shown only beside them, never on its own.

The full per class train and test interval counts are tabulated in
[docs/source/datasets.md](../source/datasets.md). The dataset is
CC BY-NC-ND, copyright Vicomtech: no frames, clips, or derived data are committed
to this repository, only the code, the metrics files, and this card. The
committed metrics files are the traceable record behind every figure here:
[distraction-mobilenet_v3_small-metrics.json](distraction-mobilenet_v3_small-metrics.json),
[distraction-efficientnet_b0-metrics.json](distraction-efficientnet_b0-metrics.json),
[distraction-mobilenet_v2-metrics.json](distraction-mobilenet_v2-metrics.json),
[distraction-shufflenet_v2_x0_5-metrics.json](distraction-shufflenet_v2_x0_5-metrics.json),
and [distraction-majority-metrics.json](distraction-majority-metrics.json).

## Method

Each candidate is a torchvision backbone pretrained on ImageNet, used as a frozen
feature extractor: the convolutional trunk is not fine tuned, its penultimate
features are cached once per image, and only a single linear head is trained on
those cached features. Freezing the trunk was a deliberate choice driven by the
training hardware, which is CPU only: caching features once and fitting a linear
head keeps training tractable without a GPU, at the cost of the accuracy a full
fine tune might have reached. Inputs are RGB frames resized to 224 by 224 with
the standard ImageNet normalization. The majority baseline in the tables predicts
the single most frequent training class for every frame and exists to show how
much of the overall accuracy is free.

## Candidate comparison

All four backbones and the majority baseline were evaluated on the same fixed
subject independent test split. Latency and throughput are the float exports
measured on a Raspberry Pi 4B at the production input shape (1, 3, 224, 224); see
[docs/perf/edge-benchmark.md](../perf/edge-benchmark.md) for the full benchmark,
including the int8 exports, which run three to five times slower on this device
and are therefore not deployed.

| Model | Balanced accuracy | Overall accuracy | Pi float mean (ms) | Pi float throughput (fps) |
|-------|-------------------|------------------|--------------------|---------------------------|
| efficientnet_b0 | 49.42% | 61.29% | 98.8 | 10.1 |
| mobilenet_v3_small | 45.10% | 64.92% | 18.3 | 54.6 |
| mobilenet_v2 | 36.29% | 65.52% | 46.6 | 21.5 |
| shufflenet_v2_x0_5 | 34.89% | 63.10% | 9.8 | 101.9 |
| majority baseline | 9.09% | 38.71% | n/a | n/a |

Read overall accuracy against the baseline: `mobilenet_v2` posts the highest
overall accuracy (65.52%) yet the second lowest balanced accuracy (36.29%),
because it wins its extra points on the majority classes while collapsing on the
rare ones. Balanced accuracy separates the models honestly, and on that measure
`efficientnet_b0` leads.

## Deployed model

The deployed backbone is **mobilenet_v3_small (float)**. `efficientnet_b0` has
the highest balanced accuracy but runs near 10 fps on the Pi, while
`mobilenet_v3_small` trails it by 4.32 balanced accuracy points (45.10% against
49.42%) at more than five times the speed (54.6 fps against 10.1 fps). For a
track that runs periodically rather than every frame, that trade favors the
faster model, so `mobilenet_v3_small` is deployed. The reasoning and the raw
latency figures live in [docs/perf/edge-benchmark.md](../perf/edge-benchmark.md).

Per class recall for the deployed model on the 11 testable classes, with the test
support count beside each, from
[distraction-mobilenet_v3_small-metrics.json](distraction-mobilenet_v3_small-metrics.json):

| Class | Recall | Test support |
|-------|--------|--------------|
| safe_drive | 96.35% | 192 |
| reach_side | 76.56% | 128 |
| reach_backseat | 75.00% | 8 |
| hair_and_makeup | 69.23% | 13 |
| drinking | 50.00% | 18 |
| phonecall_right | 50.00% | 8 |
| radio | 31.25% | 16 |
| phonecall_left | 25.00% | 8 |
| texting_right | 21.43% | 14 |
| talking_to_passenger | 1.23% | 81 |
| texting_left | 0.00% | 10 |
| change_gear | untestable | 0 |
| standstill_or_waiting | untestable | 0 |

The mean of the eleven testable recalls is the 45.10% balanced accuracy reported
above.

## Failure modes

- **The rare and chirality classes are the hardest.** `texting_left` scores 0.00%
  recall for all four candidates, and `texting_right` is barely recovered (only
  `mobilenet_v3_small` reaches it at all, at 21.43%). Left versus right hand
  activities and the low support classes (test support 8 to 14) are where every
  model is weakest, and no chirality claim should be read into these numbers.
- **`talking_to_passenger` is common but poorly detected.** It is the third most
  frequent test class (81 intervals) yet no candidate exceeds 12.35% recall on
  it (`mobilenet_v3_small` reaches only 1.23%). The frozen features do not
  separate it from `safe_drive`, into which most of its frames are misread.
- **Overall accuracy is inflated by imbalance.** With three classes covering 81%
  of the test set, a model can look strong on overall accuracy while missing most
  of the distraction classes that matter. This is exactly why balanced accuracy
  and per class recall are the reported headline.
- **Small population.** Four test subjects is a narrow basis for generalization,
  and two classes are entirely untested. These numbers characterize this split,
  not driving in general.

## Limitations and disclaimer

This is an assistive prototype, not a medical or safety of life device. It makes
no diagnostic or safety of life claim and does not decide fitness to drive. It does
not identify a person or build any face profile (see the privacy threat model
under docs/security). Every figure here traces to a committed metrics file and a
fixed, subject independent split rather than to a remembered value, and the
imbalance and 11 of 13 class coverage are stated beside the numbers so that
overall accuracy is never read on its own.
