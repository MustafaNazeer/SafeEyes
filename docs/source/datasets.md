# Datasets

Public reference for the datasets SafeEyes trains and evaluates on. Every dataset
here is obtainable by any developer without institutional sign off; no reported
claim depends on access a reader could not themselves obtain. The raw data itself
is never committed (it is large and gitignored under `data/`); this file records
where to get each dataset, how it is laid out on disk, and how the tooling turns
it into fixed, subject independent splits.

## Conventions used here

- Raw datasets live under `data/` (gitignored). Nothing large or raw is tracked.
- Split manifests are written to `splits/` (tracked). A manifest is a small CSV of
  (sample_id, subject_id, label) plus a `summary.json` recording the seed and the
  class distribution; datasets annotated with frame intervals extend the row with
  (start_frame, end_frame). These manifests are the fixed record every reported
  metric traces back to.
- Subject independence is mandatory: no subject may appear in more than one split.
  The split tooling enforces this and fails loudly if it is ever violated.

## UTA-RLDD (Real Life Drowsiness Dataset)

- **Use:** primary labels for the temporal fatigue classifier.
- **Contents:** roughly 30 hours of RGB video, 180 clips (60 subjects, one clip per
  class), three classes. The class is encoded in the clip file name: `0` is alert,
  `5` is low vigilance, `10` is drowsy. The 60 subjects are organized into five folds
  of twelve.
- **Source:** project page at https://sites.google.com/view/utarldd/home, which
  distributes the data through Google Drive. That source carries a per file download
  quota that can block automated retrieval, so the community Kaggle mirror
  (`rishab260/uta-reallife-drowsiness-dataset`, CC0-1.0) is used instead.
- **Coverage of the mirror used:** the Kaggle mirror carries folds 1 to 4 only, that
  is 48 of the 60 subjects (141 clips, 47 per class), not the full five folds. The
  reported temporal results are on these 48 subjects. Subject numbering is global, so
  no subject is split across folds and the subject independent property holds. The
  mirror nests each clip one level deeper than the canonical layout
  (`Fold{N}_part{M}/Fold{N}_part{M}/<subjectID>/{0,5,10}.<ext>`, with a mix of `.mov`,
  `.MOV`, and `.mp4`), which the recursive split tooling handles transparently.
- **Expected layout** (point `--root` at the directory that holds the fold folders):

  ```
  data/uta-rldd/
    Fold1/<subjectID>/{0,5,10}.mp4
    Fold2/<subjectID>/{0,5,10}.mp4
    ...
  ```

  The split tooling treats each clip as one sample, keyed on its subject (the parent
  folder), so all of a subject's clips always land in the same bucket.

## MRL Eye Dataset

- **Use:** training and evaluation for the open or closed eye state classifier.
- **Contents:** roughly 84,900 infrared eye images, captured under varied lighting and
  three sensors. Each image filename encodes its annotations as underscore separated
  fields, in the order documented by the dataset authors:

  ```
  subjectID_imageID_gender_glasses_eyeState_reflections_lighting_sensorID.png
  ```

  The fifth field is eye state: `0` is closed, `1` is open. The first field is the
  subject, which the split keys on so the classifier never sees the same subject in
  both training and evaluation.
- **Source:** official page at https://mrl.cs.vsb.cz/eyedataset.html, with community
  Kaggle mirrors available (for example `imadeddinedjerarda/mrl-eye-dataset`). Confirm
  any mirror preserves the original filenames, since the labels live in the names.
- **Note:** the infrared imagery is useful groundwork for a future low light extension.

## YawDD (Yawning Detection Dataset)

- **Use:** held out validation of the mouth aspect ratio yawn signal (second
  version work in development; no reported number uses YawDD yet).
- **Contents:** roughly 351 driver videos in real, varying illumination, from
  frontal and mirror camera positions, covering yawning, talking, and normal
  driving.
- **Source:** IEEE DataPort at
  https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset. Open
  access on a free account login; access is registered and held by the author.
  Before any YawDD derived number is published, the working copy's file inventory
  is checked against the official distribution listing and re downloaded from the
  official source if they differ.

## DMD (Driver Monitoring Dataset)

- **Use:** second version work in development: distraction activity classification
  (distraction bundle), gaze zone estimation (gaze bundle), and cross dataset
  evaluation of the temporal fatigue classifier (drowsiness bundle). No reported
  number uses DMD yet; this entry records provenance and terms ahead of that work.
- **Contents:** RGB video bundles from the Driver Monitoring Dataset by Vicomtech:
  distraction (drivers performing activities such as texting, operating the radio,
  and drinking), gaze (gaze zone material), and drowsiness. Each bundle ships with
  its own README and a SHA256 checksum manifest; verify after download.
- **Source:** https://dmd.vicomtech.org/, distributed through per bundle request
  forms that grant download links to any requester. Only RGB material is currently
  distributed by the authors.
- **License:** Creative Commons Attribution NonCommercial NoDerivatives 4.0
  (CC BY-NC-ND 4.0), copyright Vicomtech. Use here is noncommercial research with
  attribution. No frames, clips, or transformed copies of the material are ever
  redistributed or committed; raw data lives only under `data/dmd/` (gitignored).
- **Distraction bundle, on disk layout:** only the annotations and the body camera
  videos are extracted from the bundle archives, into the gitignored data tree:

  ```
  data/dmd/distraction/
    <group>/<subject>/<session>/
      <stem>_rgb_body.mp4
      <stem>_rgb_ann_distraction.json
  ```

  Each session pairs one body camera video (the wide cabin view, the right choice
  for activity classification) with one OpenLABEL annotation file that labels
  driver actions as frame intervals. The working copy holds 14 subjects across 49
  sessions.
- **Interval samples:** temporally annotated video is modeled as one sample per
  labeled action interval rather than one sample per clip. A sample id is the video
  path relative to the dataset root plus the frame range, for example
  `gA/1/s1/<stem>_rgb_body.mp4#123-456`, and manifest rows carry the interval
  columns described under the conventions above. The rationale is recorded in
  [ADR 0005](../adr/0005-interval-samples-for-temporally-annotated-video.md).
- **Verified action taxonomy:** the manifest builder recognizes exactly the
  thirteen `driver_actions` types found by scanning every annotation file in the
  working copy. `unclassified` intervals are skipped, and an action type outside
  the pinned set fails the build loudly instead of passing silently, so a future
  bundle revision cannot slip new classes in unnoticed.
- **Split:** subject level, seed 0, ratios 0.7 train, 0.0 validation, 0.3 test:
  10 train and 4 test subjects out of 14, no subject overlap, 1,500 train and 496
  test interval samples. Per class interval counts from
  `splits/dmd-distraction/summary.json`:

  | Class | Train | Test |
  |---|---|---|
  | change_gear | 5 | 0 |
  | drinking | 62 | 18 |
  | hair_and_makeup | 58 | 13 |
  | phonecall_left | 24 | 8 |
  | phonecall_right | 25 | 8 |
  | radio | 64 | 16 |
  | reach_backseat | 24 | 8 |
  | reach_side | 394 | 128 |
  | safe_drive | 587 | 192 |
  | standstill_or_waiting | 4 | 0 |
  | talking_to_passenger | 186 | 81 |
  | texting_left | 34 | 10 |
  | texting_right | 33 | 14 |

  Honest caveats, stated up front: 14 subjects is a small population, so any future
  accuracy on this split must always be shown beside these per class counts. The
  two rare classes, `change_gear` and `standstill_or_waiting`, appear in only three
  sessions each and landed entirely in train, so the test split covers 11 of the 13
  classes. Class imbalance is heavy (`safe_drive` and `reach_side` dominate). No
  reported model number exists for this dataset yet.
- **Frame extraction:** training images are produced from the tracked manifests
  with the split and extraction tools:

  ```
  safeeyes-build-splits --dataset dmd-distraction --root data/dmd/distraction \
    --out splits/dmd-distraction --ratios 0.7 0.0 0.3
  safeeyes-extract-interval-frames --manifest splits/dmd-distraction/train.csv \
    splits/dmd-distraction/test.csv --video-root data/dmd/distraction \
    --out-root data/dmd/distraction-frames
  ```

  The extractor decodes each video once no matter how many intervals it carries,
  writes frames only under the gitignored data tree, is resumable, and fails
  loudly on a missing video, a duplicate sample id, or a video that ends before
  yielding every requested frame.

## Integrity verification

After downloading an archive, record its SHA256 and verify it before unpacking, so the
provenance above matches the bytes on disk. The helper used for this is
`safeeyes.data.checksums`:

```python
from safeeyes.data.checksums import sha256_of_file
print(sha256_of_file("data/downloads/mrl-eye-dataset.zip"))
```

Keep the recorded hashes alongside the data so a re download or a swapped file is
caught.

## Building subject independent splits

Once a dataset is on disk, generate its fixed split manifests with the build tool. It
walks the dataset into a manifest, partitions it by subject, verifies subject
independence, and writes the manifests plus a summary:

```
python -m safeeyes.data.build_splits --dataset uta-rldd --root data/uta-rldd --out splits/uta-rldd
python -m safeeyes.data.build_splits --dataset mrl      --root data/mrl-eye  --out splits/mrl-eye --ratios 0.8 0.0 0.2
```

The default ratios are 0.7 train, 0.15 validation, 0.15 test, applied at the subject
level. A validation ratio of 0 produces a two way train and test split, which is the
common arrangement for the eye state classifier. The split is deterministic for a given
seed, so the same command always reproduces the same partition.

## Class balance and handling

Class balance differs by dataset and is reported, never hidden:

- **UTA-RLDD** is balanced by construction: each subject contributes exactly one clip
  per class, so a subject level split stays class balanced across train, validation,
  and test without any extra effort.
- **MRL Eye** is roughly balanced between open and closed overall, but the counts vary
  per subject. Because the split is made at the subject level (the correctness
  requirement), the per split open/closed ratio can drift from the global ratio. The
  `summary.json` written next to every split records the exact class distribution per
  bucket so any imbalance is visible. Inverse-frequency class weighting is available at
  training time but is off by default: it was evaluated and found to lower held out
  accuracy on the MRL split, so the imbalance is surfaced through balanced accuracy and
  per class recall rather than corrected by weighting. Subjects are never leaked across
  splits to force a balance.

## Datasets deliberately not used

Institution gated datasets (for example NTHU-DDD and DGW) require signed data use
agreements scoped to a specific institution and team. They are permanently
excluded: no claim in this project may depend on access an unaffiliated developer
cannot obtain, and material obtained under an institution scoped agreement never
flows into this project.
