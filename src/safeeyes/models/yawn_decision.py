"""Video level decision threshold selection on a held out validation fold.

A video fires, is predicted yawning, if any of its proposed events scores at
or above tau. A video with no proposed events therefore scores 0.0 and can
never fire, which is what makes the detector inherit whatever recall ceiling
the MAR proposal stage already set.

Tau is picked by reusing ``precision_recall`` from ``yawn_baseline.py`` rather
than writing a second scoring function, so the geometric baseline and this
classifier are measured by identical code. Each video's single aggregated
score is folded into a one row mar array; calling ``precision_recall`` with
``min_duration=1`` treats a score at or above tau as exactly one detected run
of length 1, which is the same "fires or does not" rule stated above.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from safeeyes.models.yawn_events import YawnEvent
from safeeyes.temporal.yawn_baseline import precision_recall

_POSITIVE_LABEL = "Yawning"
_NEGATIVE_LABEL = "Normal"


def video_scores(events: Sequence[YawnEvent], scores: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for event, score in zip(events, scores, strict=True):
        current = result.get(event.sample_id)
        if current is None or score > current:
            result[event.sample_id] = float(score)
    return result


def select_tau(
    video_scores: dict[str, float],
    truths: dict[str, bool],
    taus: Sequence[float],
    min_recall: float = 0.90,
) -> dict[str, object]:
    videos = [
        (
            sample_id,
            _POSITIVE_LABEL if truth else _NEGATIVE_LABEL,
            np.array([video_scores.get(sample_id, 0.0)]),
        )
        for sample_id, truth in truths.items()
    ]

    measurements: list[tuple[float, float, float, int, int, int]] = []
    for tau in taus:
        precision, recall, tp, fp, fn = precision_recall(videos, tau, min_duration=1)
        measurements.append((tau, precision, recall, tp, fp, fn))

    rows: list[dict[str, object]] = [
        {"tau": tau, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}
        for tau, precision, recall, tp, fp, fn in measurements
    ]

    eligible = [m for m in measurements if m[2] >= min_recall]
    floor_met = bool(eligible)
    if floor_met:
        selected = max(eligible, key=lambda m: (m[1], -m[0]))[0]
    else:
        selected = max(measurements, key=lambda m: (m[2], -m[0]))[0]

    return {"selected": selected, "floor_met": floor_met, "rows": rows}
