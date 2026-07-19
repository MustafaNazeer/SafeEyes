"""Video level validation of the geometric MAR yawn signal.

The detection threshold is preregistered: derived from UTA train subject MAR
statistics only (the 99th percentile of all per frame MAR values across the
train split features), fixed before any YawDD data was listed, extracted, or
scored, and never revised after. YawDD is therefore a purely held out test set.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

from safeeyes.data.yawdd import is_yawning
from safeeyes.temporal.features import count_onsets


def derive_mar_threshold(mar_values: np.ndarray, percentile: float = 99.0) -> float:
    values = np.asarray(mar_values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot derive a threshold from no MAR values")
    return float(np.percentile(values, percentile))


MAR_YAWN_THRESHOLD: float = 0.616703


def video_predicts_yawning(mar: np.ndarray, threshold: float) -> bool:
    values = np.asarray(mar, dtype=float)
    if values.size == 0:
        return False
    return count_onsets(cast(Sequence[float], values), threshold, "above") >= 1


def score_videos(
    videos: Sequence[tuple[str, np.ndarray]], threshold: float
) -> dict[str, object]:
    per_category: dict[str, dict[str, int]] = {}
    tp = fp = fn = 0
    talking_n = talking_fp = 0
    for label, mar in videos:
        actions = label.split("&")
        truth = is_yawning(actions)
        predicted = video_predicts_yawning(mar, threshold)
        entry = per_category.setdefault(label, {"n": 0, "predicted_yawning": 0})
        entry["n"] += 1
        entry["predicted_yawning"] += int(predicted)
        if truth and predicted:
            tp += 1
        elif truth:
            fn += 1
        elif predicted:
            fp += 1
        if not truth and "Talking" in actions:
            talking_n += 1
            talking_fp += int(predicted)
    return {
        "n_videos": len(videos),
        "n_yawning_true": tp + fn,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "per_category": per_category,
        "talking_false_positive_rate": talking_fp / talking_n if talking_n else None,
    }


def threshold_curve(
    videos: Sequence[tuple[str, np.ndarray]], thresholds: Sequence[float]
) -> list[dict[str, object]]:
    curve = []
    for threshold in thresholds:
        s = score_videos(videos, threshold)
        curve.append(
            {"threshold": float(threshold), "precision": s["precision"], "recall": s["recall"]}
        )
    return curve
