"""Subject independent split for the DMD gaze bundle.

Fourteen subjects split 8 train, 2 validation, 4 test, following the ratio the
distraction bundle used on its own 14 subjects. The validation fold exists so
the deploy bar can be fixed without ever reading the test split, which is
scored exactly once.

The split is a function of the seed and the subject set alone, never of the
order they were listed in, so the manifests regenerate byte identically. That
matters because DMD is CC BY-NC-ND: the manifests are not redistributed, so a
reader reproduces them from this builder rather than downloading them.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

N_TRAIN = 8
N_VAL = 2
N_TEST = 4
N_SUBJECTS = N_TRAIN + N_VAL + N_TEST


def build_gaze_split(subjects: Sequence[str], seed: int = 0) -> dict[str, list[str]]:
    listed = list(subjects)
    unique = set(listed)
    if len(listed) != len(unique):
        duplicates = sorted({s for s in listed if listed.count(s) > 1})
        raise ValueError(f"duplicate subjects in the corpus: {duplicates}")
    if len(unique) != N_SUBJECTS:
        raise ValueError(f"expected {N_SUBJECTS} subjects, got {len(unique)}")

    shuffled = sorted(unique)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": sorted(shuffled[:N_TRAIN]),
        "val": sorted(shuffled[N_TRAIN : N_TRAIN + N_VAL]),
        "test": sorted(shuffled[N_TRAIN + N_VAL :]),
    }
