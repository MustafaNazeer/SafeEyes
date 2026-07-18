"""The driver monitor pipeline fuses the fatigue and distraction tracks.

It drives the existing drowsiness pipeline from the per frame features and the
distraction track from the raw frame, then reports the fused tier (the max of the
two) alongside each sub-tier. These tests use stubs so the fusion logic is
exercised without a camera or a model.
"""

from __future__ import annotations

import numpy as np

from safeeyes.alert.monitor import DriverMonitorPipeline
from safeeyes.alert.state_machine import AlertTier, DistractionAlertTrack
from safeeyes.distraction.scheduler import DistractionState


class _StubDrowsiness:
    def __init__(self, tier: AlertTier, level: int) -> None:
        self._tier = tier
        self._level = level

    def process(self, features: np.ndarray) -> AlertTier:
        return self._tier

    @property
    def fatigue_level(self) -> int:
        return self._level


class _StubScheduler:
    def __init__(self, state: DistractionState) -> None:
        self._state = state

    def update(self, frame: object) -> DistractionState:
        return self._state


def _distracted_state() -> DistractionState:
    probs = np.array([0.1, 0.8, 0.1], dtype=np.float32)
    return DistractionState(probs, "texting_right", True, True)


def test_fused_tier_is_the_max_and_subtiers_are_exposed() -> None:
    monitor = DriverMonitorPipeline(
        _StubDrowsiness(AlertTier.ALARM, 2),
        _StubScheduler(_distracted_state()),
        DistractionAlertTrack(escalate_steps=1),
    )
    result = monitor.process(np.zeros(5, dtype=np.float32), frame="f")

    assert result.fatigue_tier == AlertTier.ALARM
    assert result.distraction_tier == AlertTier.VISUAL
    assert result.tier == AlertTier.ALARM
    assert result.fatigue_level == 2
    assert result.distraction.activity == "texting_right"


def test_distraction_can_raise_the_overall_tier_above_fatigue() -> None:
    monitor = DriverMonitorPipeline(
        _StubDrowsiness(AlertTier.NONE, 0),
        _StubScheduler(_distracted_state()),
        DistractionAlertTrack(escalate_steps=1),
    )
    result = monitor.process(np.zeros(5, dtype=np.float32), frame="f")

    assert result.fatigue_tier == AlertTier.NONE
    assert result.distraction_tier == AlertTier.VISUAL
    assert result.tier == AlertTier.VISUAL
