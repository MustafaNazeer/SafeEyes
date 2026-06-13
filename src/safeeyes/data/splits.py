"""Subject independent dataset splitting.

The hard guarantee this module enforces is that no subject ever appears in more
than one split. Same subject leakage between train and test silently inflates
reported metrics, so every split produced here keeps all of a subject's samples
together in a single bucket. Splits are deterministic given a seed so that every
reported number traces back to a reproducible partition.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    sample_id: str
    subject_id: str
    label: str


@dataclass(frozen=True)
class Split:
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]


class SubjectLeakageError(AssertionError):
    """Raised when a subject is found in more than one split."""


def subject_independent_split(
    samples: Sequence[Sample],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> Split:
    by_subject: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_subject[sample.subject_id].append(sample)

    subjects = sorted(by_subject)
    random.Random(seed).shuffle(subjects)

    n = len(subjects)
    n_train = round(ratios[0] * n)
    n_val = round(ratios[1] * n)
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train : n_train + n_val]
    test_subjects = subjects[n_train + n_val :]

    def gather(chosen: list[str]) -> list[Sample]:
        return [sample for subject in chosen for sample in by_subject[subject]]

    return Split(
        train=gather(train_subjects),
        val=gather(val_subjects),
        test=gather(test_subjects),
    )


def class_distribution(samples: Sequence[Sample]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[sample.label] += 1
    return dict(counts)


def assert_subject_independent(split: Split) -> None:
    train_subjects = {s.subject_id for s in split.train}
    val_subjects = {s.subject_id for s in split.val}
    test_subjects = {s.subject_id for s in split.test}
    leaked = (
        (train_subjects & val_subjects)
        | (train_subjects & test_subjects)
        | (val_subjects & test_subjects)
    )
    if leaked:
        raise SubjectLeakageError(f"subjects appear in multiple splits: {sorted(leaked)}")
