"""Binary drowsy cross-dataset evaluation of the v1 temporal model.

The v1 model predicts three UTA fatigue levels; on DMD we collapse its output to
drowsy versus not and score it against frame-level drowsy labels windowed the same
way the model consumes them. These tests pin the window-to-label alignment, the
positive-fraction threshold, and the binary metric arithmetic with a stub
classifier so no trained model or dataset is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from safeeyes.temporal.cross_eval import evaluate_binary_drowsy, windowed_binary_labels


def test_window_labels_align_with_feature_windows() -> None:
    feats = np.arange(300 * 2, dtype=float).reshape(300, 2)
    row = np.zeros(300, dtype=bool)
    row[150:170] = True  # a 20-frame drowsy region

    windows, labels = windowed_binary_labels(feats, row, 150, 75, positive_fraction=0.1)

    assert len(windows) == 3  # starts 0, 75, 150
    # window 0 (0..149) has no drowsy; 1 (75..224) and 2 (150..299) include 20/150 = 0.13
    assert list(labels) == [False, True, True]


def test_positive_fraction_threshold_controls_the_label() -> None:
    feats = np.zeros((150, 2))
    row = np.zeros(150, dtype=bool)
    row[:10] = True  # 10/150 = 0.067 of the window

    _, lenient = windowed_binary_labels(feats, row, 150, 150, positive_fraction=0.05)
    _, strict = windowed_binary_labels(feats, row, 150, 150, positive_fraction=0.1)

    assert lenient[0] is np.True_ or bool(lenient[0]) is True
    assert bool(strict[0]) is False


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        windowed_binary_labels(np.zeros((100, 2)), np.zeros(90, dtype=bool), 50, 25, 0.1)


def test_binary_metrics_arithmetic() -> None:
    windows = [np.zeros((150, 5)) for _ in range(4)]
    window_true = [True, True, False, False]
    preds = iter([2, 0, 2, 0])  # predicted classes; drowsy is class 2

    metrics = evaluate_binary_drowsy(windows, window_true, lambda w: next(preds))

    # TP=w0, FN=w1, FP=w2, TN=w3
    assert metrics["n_windows"] == 4
    assert metrics["n_drowsy_true"] == 2
    assert metrics["drowsy_recall"] == 0.5
    assert metrics["false_alarm_rate"] == 0.5
    assert metrics["overall_accuracy"] == 0.5
