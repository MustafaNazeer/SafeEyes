import pytest

from safeeyes.data.gaze_splits import N_TEST, N_TRAIN, N_VAL, build_gaze_split

SUBJECTS = [
    "gA-1",
    "gA-5",
    "gB-6",
    "gB-7",
    "gB-9",
    "gB-10",
    "gC-13",
    "gC-14",
    "gE-28",
    "gE-29",
    "gF-23",
    "gZ-33",
    "gZ-36",
    "gZ-37",
]


def test_split_sizes_are_eight_two_four():
    split = build_gaze_split(SUBJECTS, seed=0)
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (8, 2, 4)
    assert (N_TRAIN, N_VAL, N_TEST) == (8, 2, 4)


def test_no_subject_appears_in_two_buckets():
    split = build_gaze_split(SUBJECTS, seed=0)
    everything = split["train"] + split["val"] + split["test"]
    assert len(everything) == len(set(everything)) == 14


def test_every_input_subject_is_placed():
    split = build_gaze_split(SUBJECTS, seed=0)
    placed = set(split["train"]) | set(split["val"]) | set(split["test"])
    assert placed == set(SUBJECTS)


def test_split_is_deterministic_for_a_seed():
    assert build_gaze_split(SUBJECTS, seed=0) == build_gaze_split(SUBJECTS, seed=0)


def test_split_does_not_depend_on_input_order():
    """Regeneration must be byte identical however the subjects were listed."""
    forward = build_gaze_split(SUBJECTS, seed=0)
    backward = build_gaze_split(list(reversed(SUBJECTS)), seed=0)
    assert forward == backward


def test_a_different_seed_gives_a_different_split():
    assert build_gaze_split(SUBJECTS, seed=0) != build_gaze_split(SUBJECTS, seed=1)


def test_too_few_subjects_is_rejected():
    with pytest.raises(ValueError, match="14"):
        build_gaze_split(SUBJECTS[:5], seed=0)


def test_duplicate_subjects_are_rejected_rather_than_silently_deduplicated():
    """Silently deduplicating would hide a corpus that listed a subject twice."""
    with pytest.raises(ValueError, match="duplicate"):
        build_gaze_split([*SUBJECTS, "gA-1"], seed=0)
