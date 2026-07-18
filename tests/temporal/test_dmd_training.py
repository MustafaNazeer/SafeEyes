import numpy as np

from safeeyes.temporal.dmd_training import (
    ALERT_CLASS,
    DROWSY_CLASS,
    dmd_subject_key,
    dmd_subject_split,
    dmd_window_dataset,
)

DMD_STEMS = [
    "gA_1_s5_a", "gA_5_s5_a", "gB_10_s5_a", "gB_10_s5_b", "gB_6_s5_a", "gB_7_s5_a",
    "gB_9_s5_a", "gC_13_s5_a", "gC_14_s5_a", "gE_29_s5_a", "gF_23_s5_a", "gF_23_s5_b",
    "gZ_33_s5_a", "gZ_33_s5_b", "gZ_36_s5_a", "gZ_37_s5_a",
]


def test_dmd_subject_key_is_group_and_number() -> None:
    assert dmd_subject_key("gA_1_s5_2019-03-14T14;26;17+01;00") == "gA_1"
    assert dmd_subject_key("gB_10_s5_2019-03-13T14;17;28+01;00") == "gB_10"


def test_dmd_subject_split_is_subject_independent_and_deterministic() -> None:
    train, test = dmd_subject_split(DMD_STEMS, ratios=(0.7, 0.0, 0.3), seed=0)
    # Every recording lands in exactly one side.
    assert sorted(train + test) == sorted(DMD_STEMS)
    train_subjects = {dmd_subject_key(s) for s in train}
    test_subjects = {dmd_subject_key(s) for s in test}
    # 13 distinct subjects, split 9 train / 4 test, with no subject on both sides.
    assert len(train_subjects) == 9
    assert len(test_subjects) == 4
    assert train_subjects.isdisjoint(test_subjects)
    # Both recordings of a two-recording subject stay together.
    for subject in ("gB_10", "gF_23", "gZ_33"):
        sides = {subject in train_subjects, subject in test_subjects}
        assert sides == {True, False}
    # Deterministic.
    assert dmd_subject_split(DMD_STEMS, ratios=(0.7, 0.0, 0.3), seed=0) == (train, test)


def test_dmd_window_dataset_labels_drowsy_and_alert_windows() -> None:
    # Eight frames, first half a sustained closure (drowsy), second half awake.
    features = np.arange(16, dtype=float).reshape(8, 2)
    row_drowsy = np.array([True, True, True, True, False, False, False, False])
    x, y = dmd_window_dataset(features, row_drowsy, window_size=4, stride=2, positive_fraction=0.5)
    # Windows start at 0, 2, 4. Their drowsy fractions are 1.0, 0.5, 0.0.
    assert x.shape == (3, 4, 2)
    assert list(y) == [DROWSY_CLASS, DROWSY_CLASS, ALERT_CLASS]
    # The windows are the raw feature slices, unchanged.
    assert np.array_equal(x[0], features[0:4])
    assert np.array_equal(x[2], features[4:8])


def test_dmd_window_dataset_empty_when_shorter_than_window() -> None:
    features = np.zeros((3, 5), dtype=float)
    row_drowsy = np.zeros(3, dtype=bool)
    x, y = dmd_window_dataset(features, row_drowsy, window_size=4, stride=2, positive_fraction=0.1)
    assert x.shape == (0, 4, 5)
    assert y.shape == (0,)
