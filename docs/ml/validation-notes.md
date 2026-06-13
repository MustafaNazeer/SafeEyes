# Validation notes

Methodology review for the SafeEyes data foundation. This document records what
has been verified about the dataset tooling that turns raw downloads into fixed,
subject independent split manifests, and signs off only on what currently exists.
The split methodology, the dataset label conventions, and the class balance
handling are reviewed here. Metric level review is deferred (see the scope note at
the end).

For dataset provenance, layout, and license details, see
[docs/source/datasets.md](../source/datasets.md). For the system overview, see
[docs/architecture.md](../architecture.md).

## Why subject independence matters

Drowsiness and eye state are highly subject specific. A model that has seen any
frame of a given person during training can recognize that person at evaluation
time and score well for the wrong reason. If the same subject appears in both the
training and the evaluation bucket, every reported number is inflated and tells
you nothing about how the system generalizes to a new driver. The whole point of a
subject independent split is to make the evaluation answer the only question that
matters: how well does this work on a person it has never seen.

For this reason the split is decided at the subject level, before any frame or
window is extracted. The unit that gets partitioned is the subject, not the clip
and not the frame, so there is no path by which one person's data can land on both
sides of the split.

## Split methodology

The splitter keys every sample on its subject id and keeps all of a subject's
samples in a single bucket. The procedure is:

1. Group every sample by its subject id.
2. Sort the unique subject ids, then shuffle that list with a seeded random
   generator. Sorting first makes the input order irrelevant, so the partition
   depends only on the seed and the set of subjects, not on filesystem walk order.
3. Slice the shuffled subject list into train, validation, and test by the
   configured ratios (default 0.7, 0.15, 0.15, applied at the subject level). A
   validation ratio of zero produces a two way train and test split, which is the
   common arrangement for the eye state classifier.
4. Gather each subject's samples into its assigned bucket.

Because the split is made over subjects and then samples are gathered per subject,
a subject is structurally incapable of spanning two buckets. The split is also
deterministic: the same seed and the same set of subjects always reproduce the
same partition. The seed is recorded in the `summary.json` written next to every
split, so any published number traces back to the exact partition that produced
it.

### Manifests are the fixed record

A split is written as three small CSV manifests (train, validation, test), each a
list of (sample_id, subject_id, label), plus a `summary.json` that records the
seed and the per bucket sample count, subject count, and class distribution. The
heavy raw data is never tracked; only these manifests are. This is what makes a
reported metric reproducible: the manifest pins down exactly which samples were in
which bucket.

## Subject independence is enforced and verified

The guarantee is enforced in two places and, importantly, it is tested by
construction rather than merely asserted in prose.

- **Enforced in the splitter.** Partitioning over subjects (not samples) means a
  subject cannot be split across buckets by construction.
- **Enforced again in the build pipeline.** The build entry point runs an explicit
  subject independence check on the produced split and fails loudly if any subject
  is found in more than one bucket. This is a belt and suspenders check: it would
  catch a regression in the splitter before any manifest is written.
- **Verified in tests.** The test suite proves the property directly rather than
  trusting it. There are tests that:
  - assert the train, validation, and test subject sets are pairwise disjoint on a
    real split;
  - assert that every sample is assigned exactly once, with no duplicates and none
    dropped;
  - assert that all of a subject's samples stay together in one bucket;
  - construct a deliberately leaked split (the same subject placed in both train
    and test) and confirm the check raises rather than passing silently;
  - confirm the same seed reproduces an identical partition and that different
    seeds produce different partitions.

  Both dataset builders also have end to end tests that walk a synthetic on disk
  tree, build the split through the real build entry point, and assert subject
  independence on the result.

The full data foundation test suite passes (46 tests at the time of this review).

## Label conventions confirmed against each dataset

### MRL Eye Dataset

The MRL filename convention is, in the order documented by the dataset authors:

```
subjectID_imageID_gender_glasses_eyeState_reflections_lighting_sensorID.ext
```

The parser splits the filename stem on underscores, requires exactly eight fields,
and maps them positionally. Confirmed against the documented convention:

- The subject id is the **first** field and is what the split keys on.
- Eye state is the **fifth** field. A value of `0` is closed and `1` is open. The
  parser exposes `is_open` as `eye_state == 1`, and the sample label is `"open"`
  when the eye state is `1` and `"closed"` otherwise.

Tests cover an open eye filename (eye state `1` maps to open), a closed eye
filename (eye state `0` maps to closed), full field extraction, and a malformed
filename raising rather than silently mislabeling. Because the labels live in the
filenames, any mirror used as a source must preserve the original names; this
caveat is recorded in the dataset reference.

### UTA-RLDD (Real Life Drowsiness Dataset)

The class is encoded in the clip file name stem and maps as: `0` is alert, `5` is
low vigilance, `10` is drowsy. Confirmed against the documented convention. An
unrecognized stem raises rather than guessing a class.

The subject id is taken from the **parent folder** of each clip, matching the
documented layout where each subject is a folder holding one clip per class. The
split therefore keys on the per subject folder, so all of a subject's clips land
in the same bucket. Non video files in the tree are ignored. Tests confirm the
label mapping, that the subject id comes from the parent directory, that one
sample is produced per class clip, that non video files are skipped, and that a
built manifest splits without subject leakage.

One assumption is worth stating plainly: subject independence here relies on each
subject having a distinct folder name across the dataset. If two clips share a
parent folder name they are treated as the same subject, which is the safe
direction (it can only merge, never leak). This matches the documented UTA-RLDD
structure of distinct per subject folders.

## Class balance handling

Class balance is reported, never hidden, and never fixed by leaking subjects
across splits. The `summary.json` beside every split records the exact class
distribution per bucket, so any imbalance is visible.

- **UTA-RLDD** is balanced by construction: each subject contributes exactly one
  clip per class, so a subject level split stays balanced across buckets without
  any extra step.
- **MRL Eye** is roughly balanced between open and closed overall, but per subject
  counts vary. Because the split is made at the subject level (the correctness
  requirement comes first), the per bucket open to closed ratio can drift from the
  global ratio. This is the correct trade off: subject independence is not
  sacrificed to force a balance. Where residual imbalance matters for training, it
  is handled at training time with class weighting. The exact per bucket
  distribution is always recorded in the summary so the drift is never silent.

## Verdict and scope

Sign off on what exists: the split methodology correctly guarantees subject
independence. The guarantee is structural in the splitter, re checked in the build
pipeline, and verified directly by tests that include a deliberately leaked split.
The split is deterministic and reproducible from a recorded seed, and the produced
manifests are the fixed record that future reported numbers will trace back to.
The MRL and UTA-RLDD label conventions match their documented sources, and the
class balance handling is honest and visible in the per split summary.

No blocking findings.

**Deferred, not yet reviewed.** The geometric feature definitions (eye aspect
ratio, mouth aspect ratio, PERCLOS, head pose) and the evaluation metrics
(per class accuracy, macro AUROC, false alarm rate, detection latency) do not yet
exist in the codebase. They arrive in later work. Sign off on those definitions
and on every published metric is explicitly deferred until they are implemented
and can be reviewed against a reproducible script and the fixed splits described
here. This document signs off on the data foundation only.
