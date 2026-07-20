"""Video level validation of the geometric MAR yawn signal.

The detection threshold is preregistered: derived from UTA train subject MAR
statistics only (the 99th percentile of all per frame MAR values across the
train split features), fixed before any YawDD data was listed, extracted, or
scored, and never revised after. YawDD is therefore a purely held out test set.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from safeeyes.data.yawdd import is_yawning


def derive_mar_threshold(mar_values: np.ndarray, percentile: float = 99.0) -> float:
    values = np.asarray(mar_values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot derive a threshold from no MAR values")
    return float(np.percentile(values, percentile))


MAR_YAWN_THRESHOLD: float = 0.616703

# The number of feature rows on each side of a gated row that
# safeeyes.data.yawdd_crops guarantees a saved crop for (its margin_steps
# default), and so the maximum distance safeeyes.models.yawn_model accepts
# when substituting the nearest available crop for a missing exact row. Named
# here, in a module both already import for MAR_YAWN_THRESHOLD, because
# yawdd_crops.py is a privacy allowlisted import leaf that nothing else in the
# package may import from: the two modules cannot share this value by one
# importing it from the other, so it lives in the module they both already
# reach.
CROP_MARGIN_STEPS: int = 5


def event_runs(mar: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    values = np.asarray(mar, dtype=float)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value >= threshold and start is None:
            start = index
        elif value < threshold and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, int(values.size) - 1))
    return runs


def video_predicts_yawning(mar: np.ndarray, threshold: float, min_duration: int = 1) -> bool:
    if min_duration < 1:
        raise ValueError(f"min_duration must be positive, got {min_duration}")
    values = np.asarray(mar, dtype=float)
    if values.size == 0:
        return False
    return any(end - start + 1 >= min_duration for start, end in event_runs(values, threshold))


def score_videos(
    videos: Sequence[tuple[str, np.ndarray]], threshold: float, min_duration: int = 1
) -> dict[str, object]:
    per_category: dict[str, dict[str, int]] = {}
    tp = fp = fn = 0
    talking_n = talking_fp = 0
    for label, mar in videos:
        actions = label.split("&")
        truth = is_yawning(actions)
        predicted = video_predicts_yawning(mar, threshold, min_duration)
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
        "min_duration": min_duration,
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
