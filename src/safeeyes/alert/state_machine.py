"""Tiered, debounced alert state machine.

Consumes a stream of fatigue levels (0 alert, 1 low vigilance, 2 drowsy) and
emits an alert tier. It is built to fire fast on a sustained drowsy signal while
ignoring transient noise, and to recover slowly (hysteresis) so the alert does
not flicker. A level is only committed once it persists for a minimum number of
steps; escalation needs fewer steps than de-escalation, so the machine is quick
to warn and reluctant to stand down. Sustained committed drowsiness escalates
from an audible cue to a stronger alarm.

This is an assistive cue, not a safety of life device.
"""

from __future__ import annotations

from enum import IntEnum


class AlertTier(IntEnum):
    NONE = 0
    VISUAL = 1
    AUDIBLE = 2
    ALARM = 3


_VALID_LEVELS = (0, 1, 2)


class AlertStateMachine:
    def __init__(
        self,
        escalate_steps: int = 3,
        de_escalate_steps: int = 8,
        alarm_after: int = 15,
    ) -> None:
        if escalate_steps <= 0 or de_escalate_steps <= 0 or alarm_after <= 0:
            raise ValueError("step thresholds must be positive")
        self._escalate_steps = escalate_steps
        self._de_escalate_steps = de_escalate_steps
        self._alarm_after = alarm_after
        self.reset()

    def reset(self) -> None:
        self._committed = 0
        self._candidate: int | None = None
        self._candidate_count = 0
        self._drowsy_steps = 0

    def update(self, fatigue_level: int) -> AlertTier:
        if fatigue_level not in _VALID_LEVELS:
            raise ValueError(f"fatigue level must be one of {_VALID_LEVELS}, got {fatigue_level}")

        if fatigue_level == self._committed:
            self._candidate = None
            self._candidate_count = 0
        else:
            if fatigue_level == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate = fatigue_level
                self._candidate_count = 1
            threshold = (
                self._escalate_steps
                if fatigue_level > self._committed
                else self._de_escalate_steps
            )
            if self._candidate_count >= threshold:
                self._committed = fatigue_level
                self._candidate = None
                self._candidate_count = 0

        self._drowsy_steps = self._drowsy_steps + 1 if self._committed == 2 else 0
        return self.current_tier

    @property
    def current_tier(self) -> AlertTier:
        if self._committed == 0:
            return AlertTier.NONE
        if self._committed == 1:
            return AlertTier.VISUAL
        return AlertTier.ALARM if self._drowsy_steps >= self._alarm_after else AlertTier.AUDIBLE
