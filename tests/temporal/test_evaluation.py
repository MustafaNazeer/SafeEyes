import numpy as np
import pytest

from safeeyes.temporal.evaluation import (
    evaluate_predictions,
    false_alarm_rate,
    macro_auroc,
    overall_accuracy,
    per_class_accuracy,
)


def test_per_class_accuracy_is_recall_per_class() -> None:
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 1, 2]
    acc = per_class_accuracy(y_true, y_pred, n_classes=3)
    assert acc[0] == pytest.approx(0.5)
    assert acc[1] == pytest.approx(1.0)
    assert acc[2] == pytest.approx(1.0)


def test_overall_accuracy() -> None:
    assert overall_accuracy([0, 0, 1, 1, 2], [0, 1, 1, 1, 2]) == pytest.approx(0.8)


def test_false_alarm_rate_is_fp_over_negatives() -> None:
    # alarm class is 2 (drowsy). Negatives are true != 2.
    y_true = [0, 1, 2]
    y_pred = [2, 1, 2]  # sample 0 is a false alarm
    assert false_alarm_rate(y_true, y_pred, alarm_class=2) == pytest.approx(0.5)


def test_false_alarm_rate_no_negatives_is_zero() -> None:
    assert false_alarm_rate([2, 2], [2, 0], alarm_class=2) == 0.0


def test_macro_auroc_perfect_scores_is_one() -> None:
    y_true = [0, 1, 2]
    y_score = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    assert macro_auroc(y_true, y_score, n_classes=3) == pytest.approx(1.0)


def test_evaluate_predictions_bundles_metrics() -> None:
    y_true = [0, 1, 2, 2]
    y_pred = [0, 1, 2, 0]
    y_score = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.6, 0.1, 0.3]]
    )
    report = evaluate_predictions(y_true, y_pred, y_score, n_classes=3, alarm_class=2)
    assert "overall_accuracy" in report
    assert "macro_auroc" in report
    assert "false_alarm_rate" in report
    assert "per_class_accuracy" in report
    assert report["per_class_accuracy"]["2"] == pytest.approx(0.5)
