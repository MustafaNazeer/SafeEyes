"""Eyes off road alert track.

A third parallel track beside fatigue and distraction, fused by severity
through fuse_tiers. Glancing away from the road is normal driving: mirrors,
instruments, and shoulder checks all take the eyes off the forward view for a
moment. What distinguishes a hazard is duration, so the track fires only once
the gaze has stayed off road continuously.

Durations are wall clock seconds and the track measures them itself from an
injected clock. It deliberately knows nothing about the loop's frame rate. An
earlier version converted seconds to a frame count using a rate supplied at
construction, which is correct only while that rate is accurate: the live loop
passed a nominal 11 fps and then ran at 7 under screen capture, silently
stretching a 2 second bar to 3.1 seconds of real time. Measuring elapsed time
directly removes the caller's ability to get it wrong, which is the same class
of defect the yawn cadence trap recorded.

The track never reaches ALARM. Eyes off road is a lesser cue than sustained
drowsiness, matching how the distraction track is bounded at AUDIBLE.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from safeeyes.alert.state_machine import AlertTier


class EyesOffRoadTrack:
    def __init__(
        self,
        min_seconds: float = 2.0,
        audible_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min_seconds <= 0:
            raise ValueError(f"min_seconds must be positive, got {min_seconds}")
        if audible_seconds is not None and audible_seconds < min_seconds:
            raise ValueError(
                f"audible_seconds {audible_seconds} must not precede min_seconds {min_seconds}"
            )

        self.min_seconds = min_seconds
        self.audible_seconds = audible_seconds
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self._streak_started: float | None = None

    def update(self, off_road: bool) -> AlertTier:
        if not off_road:
            self._streak_started = None
            return AlertTier.NONE

        now = self._clock()
        if self._streak_started is None:
            self._streak_started = now
        elapsed = now - self._streak_started

        if self.audible_seconds is not None and elapsed >= self.audible_seconds:
            return AlertTier.AUDIBLE
        if elapsed >= self.min_seconds:
            return AlertTier.VISUAL
        return AlertTier.NONE
