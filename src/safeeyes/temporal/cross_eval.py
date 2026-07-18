"""Binary drowsy cross-dataset evaluation for the temporal fatigue model.

The v1 temporal model was trained on UTA-RLDD's three fatigue levels. To test how
it generalizes to a different dataset (DMD drowsiness), we collapse its prediction
to drowsy versus not and score it against DMD's frame-level drowsy labels, windowed
exactly the way the model consumes features. The window level ground truth is
drowsy when at least ``positive_fraction`` of the window's frames carry the
frame-level drowsy label, so a brief microsleep does not by itself mark a long
window drowsy unless it covers a meaningful share of it.

Any number produced here carries two caveats stated wherever it is reported: DMD
drowsiness was acted rather than genuine, and the drowsy label is a sustained eye
closure proxy, not an independent drowsiness ground truth.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from safeeyes.temporal.evaluation import (
    false_alarm_rate,
    overall_accuracy,
    per_class_accuracy,
)
from safeeyes.temporal.window import make_windows

DROWSY_CLASS = 2


def windowed_binary_labels(
    features: np.ndarray,
    row_drowsy: np.ndarray,
    window_size: int,
    stride: int,
    positive_fraction: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Slice features into windows and give each a binary drowsy label.

    The label windows use the same start and stride as the feature windows, so the
    two stay aligned. A window is drowsy when the mean of its frame-level drowsy
    flags reaches ``positive_fraction``.
    """
    feats = np.asarray(features, dtype=float)
    labels = np.asarray(row_drowsy, dtype=bool)
    if feats.shape[0] != labels.shape[0]:
        raise ValueError("features and row labels must have the same number of frames")
    windows, _ = make_windows(feats, 0, window_size, stride)
    window_labels: list[bool] = []
    start = 0
    for _ in windows:
        segment = labels[start : start + window_size]
        window_labels.append(bool(segment.mean() >= positive_fraction))
        start += stride
    return windows, np.asarray(window_labels, dtype=bool)


def evaluate_binary_drowsy(
    windows: Sequence[np.ndarray],
    window_true: Sequence[bool] | np.ndarray,
    classify: Callable[[np.ndarray], int],
    drowsy_class: int = DROWSY_CLASS,
) -> dict[str, object]:
    """Score a fatigue-level classifier as a binary drowsy detector.

    ``classify`` returns a fatigue class per window; a prediction of ``drowsy_class``
    counts as drowsy. Reports the drowsy detection rate (recall), the false alarm
    rate on genuinely not-drowsy windows, and overall accuracy.
    """
    y_true = np.asarray(window_true, dtype=bool).astype(int)
    y_pred = np.array(
        [int(classify(np.asarray(window)) == drowsy_class) for window in windows],
        dtype=int,
    )
    return {
        "n_windows": int(y_true.size),
        "n_drowsy_true": int(y_true.sum()),
        "drowsy_recall": per_class_accuracy(y_true, y_pred, 2)[1],
        "false_alarm_rate": false_alarm_rate(y_true, y_pred, alarm_class=1),
        "overall_accuracy": overall_accuracy(y_true, y_pred),
    }
