# Datasets

Public reference for the datasets SafeEyes trains and evaluates on. All three are
freely obtainable without an institutional agreement. The raw data itself is never
committed (it is large and gitignored under `data/`); this file records where to get
it, how it is laid out on disk, and how the tooling turns it into fixed, subject
independent splits.

## Conventions used here

- Raw datasets live under `data/` (gitignored). Nothing large or raw is tracked.
- Split manifests are written to `splits/` (tracked). A manifest is a small CSV of
  (sample_id, subject_id, label) plus a `summary.json` recording the seed and the
  class distribution. These manifests are the fixed record every reported metric
  traces back to.
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

- **Use:** development and validation of the yawn signal.
- **Contents:** roughly 351 driver videos in real, varying illumination, from frontal
  and mirror camera positions.
- **Source:** IEEE DataPort at
  https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset. Open access on
  a free account login.

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

Gated datasets such as NTHU-DDD and DMD require institutional data use agreements an
unaffiliated developer cannot readily obtain. No headline claim depends on them.
