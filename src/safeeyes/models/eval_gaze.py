"""Evaluation for the gaze zone classifier, at frame and at interval level.

Frames inside one annotation interval are near duplicates of a single sustained
glance, so a frame level rate overstates its own confidence by the number of
frames per interval, which runs from about 100 to 230 here. The interval is the
independent unit, and both numbers are reported side by side: a model that
looks strong per frame and weak per interval is being flattered by duplication.

Every rate is reported beside its denominator. The forward gaze class carries
only 51 intervals across the whole corpus, so a bare percentage would imply a
precision the data cannot support.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from safeeyes.data.dmd_gaze import is_eyes_on_road


def interval_level_predictions(
    interval_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse each interval to one true label and one majority vote prediction."""
    ids = np.asarray(interval_ids)
    if ids.size == 0:
        return np.array([], dtype=object), np.array([], dtype=object)

    true_out: list[str] = []
    pred_out: list[str] = []
    for interval in sorted(set(ids.tolist())):
        mask = ids == interval
        truths = set(np.asarray(y_true)[mask].tolist())
        if len(truths) != 1:
            raise ValueError(
                f"interval {interval} has a single true label by construction, got {sorted(truths)}"
            )
        true_out.append(truths.pop())
        votes = Counter(np.asarray(y_pred)[mask].tolist())
        pred_out.append(votes.most_common(1)[0][0])
    return np.array(true_out, dtype=object), np.array(pred_out, dtype=object)


def binary_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """Eyes off road detection and false alarm rates, with their denominators.

    A false alarm is a forward gaze called off road, which is what would wake
    the driver for no reason. Detection is an off road gaze correctly called.
    Both are conditional rates, so the skewed class balance of a staged
    protocol does not bias them; only their denominators are small.
    """
    truth_on = np.array([is_eyes_on_road(z) for z in np.asarray(y_true).tolist()])
    pred_on = np.array([is_eyes_on_road(z) for z in np.asarray(y_pred).tolist()])

    n_on = int(truth_on.sum())
    n_off = int((~truth_on).sum())
    false_alarms = int((truth_on & ~pred_on).sum())
    detected = int((~truth_on & ~pred_on).sum())

    return {
        "n_on_road": n_on,
        "n_off_road": n_off,
        "false_alarms": false_alarms,
        "detected": detected,
        "false_alarm_rate": (false_alarms / n_on) if n_on else None,
        "detection_rate": (detected / n_off) if n_off else None,
    }


def zone_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    n = int(true.size)
    correct = int((true == pred).sum())
    per_class: dict[str, dict[str, int]] = {}
    for zone in sorted(set(true.tolist())):
        mask = true == zone
        per_class[zone] = {
            "n": int(mask.sum()),
            "correct": int((pred[mask] == zone).sum()),
        }
    return {
        "n": n,
        "correct": correct,
        "accuracy": (correct / n) if n else None,
        "per_class": per_class,
    }
