import numpy as np

from safeeyes.temporal.train_temporal import train_and_evaluate, train_and_evaluate_gbt


def _constant_sequences(value: float, label: int, count: int, length: int = 8, feats: int = 3):
    return [(np.full((length, feats), value), label) for _ in range(count)]


def test_train_and_evaluate_returns_metric_keys() -> None:
    train = _constant_sequences(0.0, 0, 6) + _constant_sequences(5.0, 1, 6)
    val = _constant_sequences(0.0, 0, 3) + _constant_sequences(5.0, 1, 3)
    report = train_and_evaluate(
        train, val, n_classes=2, window_size=4, stride=4, epochs=60, lr=0.05, seed=0
    )
    for key in ("overall_accuracy", "macro_auroc", "false_alarm_rate", "per_class_accuracy"):
        assert key in report


def test_train_and_evaluate_learns_separable_classes() -> None:
    train = _constant_sequences(0.0, 0, 8) + _constant_sequences(5.0, 1, 8)
    val = _constant_sequences(0.0, 0, 4) + _constant_sequences(5.0, 1, 4)
    report = train_and_evaluate(
        train, val, n_classes=2, window_size=4, stride=4, epochs=80, lr=0.05, seed=0
    )
    assert report["overall_accuracy"] >= 0.75


def test_gbt_baseline_harness_learns_separable_classes() -> None:
    train = _constant_sequences(0.0, 0, 8) + _constant_sequences(5.0, 1, 8)
    val = _constant_sequences(0.0, 0, 4) + _constant_sequences(5.0, 1, 4)
    report = train_and_evaluate_gbt(
        train, val, n_classes=2, window_size=4, stride=4, seed=0
    )
    assert report["overall_accuracy"] >= 0.75
    assert "macro_auroc" in report
