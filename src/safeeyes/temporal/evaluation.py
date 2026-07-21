"""Evaluation metrics for the temporal fatigue classifier.

Every reported number for the temporal stage comes from here, computed on a
fixed subject independent split. Per class accuracy and macro AUROC describe how
well the three fatigue levels separate; the false alarm rate is reported beside
them, never hidden, and is defined as the fraction of genuinely not drowsy
windows that are classified as drowsy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

IntArrayLike = Sequence[int] | np.ndarray


def per_class_accuracy(
    y_true: IntArrayLike, y_pred: IntArrayLike, n_classes: int
) -> dict[int, float]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    result: dict[int, float] = {}
    for c in range(n_classes):
        mask = true == c
        total = int(np.count_nonzero(mask))
        result[c] = float(np.mean(pred[mask] == c)) if total else math.nan
    return result


def overall_accuracy(y_true: IntArrayLike, y_pred: IntArrayLike) -> float:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    return float(np.mean(true == pred)) if true.size else 0.0


def false_alarm_rate(y_true: IntArrayLike, y_pred: IntArrayLike, alarm_class: int) -> float:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    negatives = true != alarm_class
    n_negatives = int(np.count_nonzero(negatives))
    if n_negatives == 0:
        return 0.0
    false_alarms = int(np.count_nonzero(negatives & (pred == alarm_class)))
    return false_alarms / n_negatives


def macro_auroc(y_true: IntArrayLike, y_score: np.ndarray, n_classes: int) -> float:
    scores = np.asarray(y_score, dtype=float)
    labels = list(range(n_classes))
    if n_classes == 2:
        return float(roc_auc_score(y_true, scores[:, 1], labels=labels))
    return float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro", labels=labels))


def evaluate_predictions(
    y_true: IntArrayLike,
    y_pred: IntArrayLike,
    y_score: np.ndarray,
    n_classes: int,
    alarm_class: int,
) -> dict[str, object]:
    return {
        "overall_accuracy": overall_accuracy(y_true, y_pred),
        "macro_auroc": macro_auroc(y_true, y_score, n_classes),
        "false_alarm_rate": false_alarm_rate(y_true, y_pred, alarm_class),
        "per_class_accuracy": {
            str(c): acc for c, acc in per_class_accuracy(y_true, y_pred, n_classes).items()
        },
    }
