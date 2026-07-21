"""Subject independent evaluation folds for the DMD gaze bundle.

Leave one subject out rather than a single held out split. The corpus is small
in the way that matters: 14 subjects and roughly 208 annotation intervals, with
about 100 to 230 near duplicate frames inside each interval. A 4 subject test
fold would rest the eyes off road false alarm rate on roughly 14 independent
forward glances, where one misclassified glance moves the figure by about seven
percentage points.

Holding out one subject at a time and pooling the folds puts every subject in
the test position exactly once, so the same rate rests on all 51 forward
glances instead. It also makes per subject variance visible, which a single
split hides entirely.

Nothing is tuned per fold. The classifier runs at fixed hyperparameters and the
binary signal is the argmax collapsed to eyes on road versus off, so there is no
inner selection loop and no validation fold to leak through. The deploy bar is
written down before the folds are run and the pooled result is read once.

Two subjects have a second s6 recording. Folds are built on subjects, never on
recordings, so both recordings of one person always sit on the same side.
"""

from __future__ import annotations

from collections.abc import Sequence

N_SUBJECTS = 14


def subject_folds(subjects: Sequence[str]) -> list[tuple[list[str], str]]:
    """One (train_subjects, held_out_subject) pair per subject, in sorted order."""
    listed = list(subjects)
    unique = set(listed)
    if len(listed) != len(unique):
        duplicates = sorted({s for s in listed if listed.count(s) > 1})
        raise ValueError(f"duplicate subjects in the corpus: {duplicates}")
    if not unique:
        raise ValueError("cannot build folds from an empty subject list")

    ordered = sorted(unique)
    return [([s for s in ordered if s != held_out], held_out) for held_out in ordered]
