import pytest

from safeeyes.data.splits import (
    Sample,
    Split,
    SubjectLeakageError,
    assert_subject_independent,
    class_distribution,
    subject_independent_split,
)


def _samples(subject_count: int, per_subject: int = 3) -> list[Sample]:
    out: list[Sample] = []
    for s in range(subject_count):
        subject_id = f"s{s:03d}"
        for i in range(per_subject):
            label = ["alert", "low", "drowsy"][i % 3]
            out.append(Sample(sample_id=f"{subject_id}_{i}", subject_id=subject_id, label=label))
    return out


def test_no_subject_appears_in_more_than_one_split() -> None:
    split = subject_independent_split(_samples(30), seed=0)
    train_subjects = {s.subject_id for s in split.train}
    val_subjects = {s.subject_id for s in split.val}
    test_subjects = {s.subject_id for s in split.test}
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)


def test_every_sample_is_assigned_exactly_once() -> None:
    samples = _samples(30)
    split = subject_independent_split(samples, seed=0)
    assigned = [s.sample_id for s in (*split.train, *split.val, *split.test)]
    assert sorted(assigned) == sorted(s.sample_id for s in samples)
    assert len(assigned) == len(set(assigned))


def test_all_samples_of_a_subject_stay_together() -> None:
    samples = _samples(30)
    split = subject_independent_split(samples, seed=0)
    for bucket in (split.train, split.val, split.test):
        bucket_subjects = {s.subject_id for s in bucket}
        for subject in bucket_subjects:
            owned = {s.sample_id for s in samples if s.subject_id == subject}
            in_bucket = {s.sample_id for s in bucket if s.subject_id == subject}
            assert owned == in_bucket


def test_same_seed_is_deterministic() -> None:
    a = subject_independent_split(_samples(30), seed=7)
    b = subject_independent_split(_samples(30), seed=7)
    assert [s.sample_id for s in a.train] == [s.sample_id for s in b.train]
    assert [s.sample_id for s in a.test] == [s.sample_id for s in b.test]


def test_assert_subject_independent_passes_on_clean_split() -> None:
    split = subject_independent_split(_samples(30), seed=0)
    assert_subject_independent(split)


def test_assert_subject_independent_raises_on_leaked_split() -> None:
    leaked_subject = Sample(sample_id="x", subject_id="s000", label="alert")
    bad = Split(train=[leaked_subject], val=[], test=[leaked_subject])
    with pytest.raises(SubjectLeakageError):
        assert_subject_independent(bad)


def test_subjects_are_partitioned_by_ratio() -> None:
    split = subject_independent_split(_samples(100), ratios=(0.7, 0.15, 0.15), seed=0)
    assert len({s.subject_id for s in split.train}) == 70
    assert len({s.subject_id for s in split.val}) == 15
    assert len({s.subject_id for s in split.test}) == 15


def test_two_way_split_when_val_ratio_is_zero() -> None:
    split = subject_independent_split(_samples(50), ratios=(0.8, 0.0, 0.2), seed=0)
    assert split.val == []
    assert len({s.subject_id for s in split.train}) == 40
    assert len({s.subject_id for s in split.test}) == 10


def test_different_seeds_produce_different_partitions() -> None:
    a = subject_independent_split(_samples(50), seed=1)
    b = subject_independent_split(_samples(50), seed=2)
    assert {s.subject_id for s in a.test} != {s.subject_id for s in b.test}


def test_empty_input_yields_empty_splits() -> None:
    split = subject_independent_split([], seed=0)
    assert split.train == [] and split.val == [] and split.test == []


def test_class_distribution_counts_labels() -> None:
    samples = _samples(10, per_subject=3)
    dist = class_distribution(samples)
    assert dist == {"alert": 10, "low": 10, "drowsy": 10}
