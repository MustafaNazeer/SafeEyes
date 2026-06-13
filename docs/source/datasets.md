# Datasets

Public reference for the datasets SafeEyes trains and evaluates on. All three are freely obtainable without an institutional agreement. The raw data itself is never committed (it is large and gitignored); this file records where to get it and how it is used.

## UTA-RLDD (Real Life Drowsiness Dataset)

- **Use:** primary labels for the temporal fatigue classifier.
- **Contents:** roughly 30 hours of RGB video, 60 subjects, three labeled classes (alert, low vigilance, drowsy).
- **Source:** project page at https://sites.google.com/view/utarldd/home. Community mirrors exist on Kaggle if the primary source is slow.
- **Note:** splits must be subject independent. No subject may appear in both train and test.

## MRL Eye Dataset

- **Use:** training and evaluation for the open or closed eye state classifier.
- **Contents:** roughly 84,900 infrared eye images labeled by eye state, captured under varied lighting and devices.
- **Source:** official page at https://mrl.cs.vsb.cz/eyedataset.html, with a Kaggle mirror available.
- **Note:** the infrared imagery is useful groundwork for a future low light extension.

## YawDD (Yawning Detection Dataset)

- **Use:** development and validation of the yawn signal.
- **Contents:** roughly 351 driver videos in real, varying illumination, from frontal and mirror camera positions.
- **Source:** IEEE DataPort at https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset (open access on login), with a direct HTTP mirror.

## Datasets deliberately not used

Gated datasets such as NTHU-DDD and DMD require institutional data use agreements an unaffiliated developer cannot readily obtain. No headline claim depends on them.
