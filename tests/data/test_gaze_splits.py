import pytest

from safeeyes.data.gaze_splits import N_SUBJECTS, subject_folds

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


def test_there_is_one_fold_per_subject():
    folds = subject_folds(SUBJECTS)
    assert len(folds) == len(SUBJECTS) == N_SUBJECTS


def test_every_subject_is_held_out_exactly_once():
    held_out = [fold[1] for fold in subject_folds(SUBJECTS)]
    assert sorted(held_out) == sorted(SUBJECTS)
    assert len(held_out) == len(set(held_out))


def test_the_held_out_subject_never_appears_in_its_own_training_set():
    """This is the whole point: a leak here would invalidate every number."""
    for train, held_out in subject_folds(SUBJECTS):
        assert held_out not in train
        assert len(train) == len(SUBJECTS) - 1


def test_folds_do_not_depend_on_input_order():
    forward = subject_folds(SUBJECTS)
    backward = subject_folds(list(reversed(SUBJECTS)))
    assert forward == backward


def test_folds_are_deterministic():
    assert subject_folds(SUBJECTS) == subject_folds(SUBJECTS)


def test_duplicate_subjects_are_rejected_rather_than_silently_deduplicated():
    with pytest.raises(ValueError, match="duplicate"):
        subject_folds([*SUBJECTS, "gA-1"])


def test_an_empty_corpus_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        subject_folds([])


def test_a_smaller_corpus_still_produces_one_fold_each():
    """The builder is not hard coded to 14, so a subset is still usable."""
    folds = subject_folds(SUBJECTS[:5])
    assert len(folds) == 5
    assert all(len(train) == 4 for train, _ in folds)
