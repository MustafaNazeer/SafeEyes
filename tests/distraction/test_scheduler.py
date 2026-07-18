"""The distraction scheduler runs the CNN every Nth frame and smooths its output.

The distraction backbone is too heavy to run on every frame within the fps
budget, so it runs on a cadence and its class probabilities are smoothed with an
exponential moving average between runs. Between runs the last smoothed state is
held. These tests pin the cadence, the EMA arithmetic, the hold behavior, and the
binary distracted flag.
"""

from __future__ import annotations

import numpy as np
import pytest

from safeeyes.distraction.scheduler import DistractionScheduler

LABELS = ("safe_drive", "texting_right", "phonecall_left")


def _onehot(index: int) -> np.ndarray:
    vector = np.zeros(len(LABELS), dtype=np.float32)
    vector[index] = 1.0
    return vector


def test_classifier_runs_only_on_the_cadence() -> None:
    calls: list[int] = []

    def classifier(_frame: object) -> np.ndarray:
        calls.append(1)
        return _onehot(0)

    scheduler = DistractionScheduler(classifier, LABELS, every_n=3)
    ran_flags = [scheduler.update("frame").ran for _ in range(7)]

    assert ran_flags == [False, False, True, False, False, True, False]
    assert len(calls) == 2


def test_ema_smoothing_matches_hand_computation() -> None:
    outputs = iter([_onehot(1), _onehot(2), _onehot(0)])

    def classifier(_frame: object) -> np.ndarray:
        return next(outputs)

    scheduler = DistractionScheduler(classifier, LABELS, every_n=1, ema_alpha=0.5)

    first = scheduler.update("f").probabilities
    np.testing.assert_allclose(first, _onehot(1))

    second = scheduler.update("f").probabilities
    np.testing.assert_allclose(second, 0.5 * _onehot(2) + 0.5 * _onehot(1))

    third = scheduler.update("f").probabilities
    np.testing.assert_allclose(third, 0.5 * _onehot(0) + 0.5 * second)


def test_state_is_held_between_runs() -> None:
    def classifier(_frame: object) -> np.ndarray:
        return _onehot(1)

    scheduler = DistractionScheduler(classifier, LABELS, every_n=3, ema_alpha=1.0)
    scheduler.update("f")
    scheduler.update("f")
    run = scheduler.update("f")  # third call runs
    hold_a = scheduler.update("f")
    hold_b = scheduler.update("f")

    assert run.ran is True
    assert hold_a.ran is False and hold_b.ran is False
    np.testing.assert_allclose(hold_a.probabilities, run.probabilities)
    np.testing.assert_allclose(hold_b.probabilities, run.probabilities)


def test_distracted_flag_follows_argmax() -> None:
    outputs = iter([_onehot(1), _onehot(0)])

    def classifier(_frame: object) -> np.ndarray:
        return next(outputs)

    scheduler = DistractionScheduler(classifier, LABELS, every_n=1, ema_alpha=1.0)

    first = scheduler.update("f")
    assert first.activity == "texting_right"
    assert first.distracted is True

    second = scheduler.update("f")
    assert second.activity == "safe_drive"
    assert second.distracted is False


def test_pre_run_state_is_undistracted() -> None:
    scheduler = DistractionScheduler(lambda _f: _onehot(1), LABELS, every_n=5)
    state = scheduler.update("f")
    assert state.ran is False
    assert state.distracted is False
    assert state.activity == "safe_drive"


def test_validation() -> None:
    with pytest.raises(ValueError):
        DistractionScheduler(lambda _f: _onehot(0), LABELS, every_n=0)
    with pytest.raises(ValueError):
        DistractionScheduler(lambda _f: _onehot(0), LABELS, ema_alpha=0.0)
    with pytest.raises(ValueError):
        DistractionScheduler(lambda _f: _onehot(0), LABELS, ema_alpha=1.5)
    with pytest.raises(ValueError):
        DistractionScheduler(lambda _f: _onehot(0), LABELS, safe_label="not_a_label")
