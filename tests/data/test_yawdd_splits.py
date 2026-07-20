import csv

import pytest

from safeeyes.data.splits import Sample, SubjectLeakageError, assert_subject_independent
from safeeyes.data.yawdd_splits import build_mirror_split, carve_validation


def _manifest(tmp_path, n_subjects):
    path = tmp_path / "mirror.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "subject_id", "label"])
        for i in range(n_subjects):
            for label in ("Normal", "Talking", "Yawning"):
                sample_id = f"Female_mirror/{i}-FemaleNoGlasses-{label}.avi"
                writer.writerow([sample_id, f"Female{i}", label])
    return path


def test_split_is_seventy_twenty_on_ninety_subjects(tmp_path):
    split = build_mirror_split(_manifest(tmp_path, 90), tmp_path, seed=0)
    assert len({s.subject_id for s in split.train}) == 70
    assert len({s.subject_id for s in split.test}) == 20
    assert split.val == []


def test_split_has_no_subject_leakage(tmp_path):
    split = build_mirror_split(_manifest(tmp_path, 90), tmp_path, seed=0)
    assert_subject_independent(split)


def test_leaked_split_is_caught(tmp_path):
    shared = Sample("a.avi", "Female1", "Yawning")
    from safeeyes.data.splits import Split

    with pytest.raises(SubjectLeakageError):
        assert_subject_independent(Split(train=[shared], val=[], test=[shared]))


def test_split_is_deterministic(tmp_path):
    first = build_mirror_split(_manifest(tmp_path, 90), tmp_path / "a", seed=0)
    second = build_mirror_split(_manifest(tmp_path, 90), tmp_path / "b", seed=0)
    assert [s.sample_id for s in first.test] == [s.sample_id for s in second.test]


def test_validation_is_carved_from_train_only(tmp_path):
    split = build_mirror_split(_manifest(tmp_path, 90), tmp_path, seed=0)
    inner, val = carve_validation(split.train, seed=0)
    assert len({s.subject_id for s in inner}) == 56
    assert len({s.subject_id for s in val}) == 14
    assert {s.subject_id for s in val}.isdisjoint({s.subject_id for s in split.test})
