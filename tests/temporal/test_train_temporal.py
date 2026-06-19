import numpy as np
import pytest

import safeeyes.temporal.model as model_mod
from safeeyes.temporal.train_temporal import (
    standardize_with_train_stats,
    train_and_evaluate,
    train_and_evaluate_gbt,
)


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


def test_standardize_centers_train_and_uses_train_stats_on_val() -> None:
    # two windows, one timestep, two features on very different scales
    x_train = np.array([[[0.0, 10.0]], [[2.0, 30.0]]])
    x_val = np.array([[[1.0, 20.0]]])  # equals the per-feature train mean
    xt, xv = standardize_with_train_stats(x_train, x_val)
    assert np.allclose(xt.mean(axis=(0, 1)), [0.0, 0.0], atol=1e-6)
    assert np.allclose(xt.std(axis=(0, 1)), [1.0, 1.0], atol=1e-6)
    # val is transformed with train statistics, so a val sample at the train mean maps to 0
    assert np.allclose(xv[0, 0], [0.0, 0.0], atol=1e-6)


def test_standardize_handles_a_constant_feature_without_dividing_by_zero() -> None:
    x_train = np.array([[[5.0]], [[5.0]]])
    x_val = np.array([[[5.0]]])
    xt, xv = standardize_with_train_stats(x_train, x_val)
    assert np.all(np.isfinite(xt))
    assert np.all(np.isfinite(xv))


def test_train_and_evaluate_never_forwards_more_than_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _constant_sequences(0.0, 0, 10) + _constant_sequences(5.0, 1, 10)
    val = _constant_sequences(0.0, 0, 2) + _constant_sequences(5.0, 1, 2)

    seen: list[int] = []
    original = model_mod.TemporalGRU.forward

    def spy_forward(self: model_mod.TemporalGRU, x: object) -> object:
        seen.append(x.shape[0])  # type: ignore[attr-defined]
        return original(self, x)  # type: ignore[arg-type]

    monkeypatch.setattr(model_mod.TemporalGRU, "forward", spy_forward)
    train_and_evaluate(
        train, val, n_classes=2, window_size=4, stride=4, epochs=1, lr=0.05, seed=0, batch_size=8
    )
    # The whole point of the safeguard: no single forward pass exceeds the batch
    # size, so peak activation memory is bounded regardless of dataset size.
    assert seen
    assert max(seen) <= 8


def test_train_and_evaluate_minibatches_and_still_learns() -> None:
    train = _constant_sequences(0.0, 0, 8) + _constant_sequences(5.0, 1, 8)
    val = _constant_sequences(0.0, 0, 4) + _constant_sequences(5.0, 1, 4)
    report = train_and_evaluate(
        train, val, n_classes=2, window_size=4, stride=4, epochs=80, lr=0.05, seed=0, batch_size=4
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
