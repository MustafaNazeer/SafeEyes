"""Eyes off road alert track.

A third parallel track beside fatigue and distraction, fused by severity
through fuse_tiers. Glancing away from the road is normal driving: mirrors,
instruments, and shoulder checks all take the eyes off the forward view for a
moment. What distinguishes a hazard is duration, so the track fires only once
the gaze has stayed off road continuously.

Durations are stored in **seconds** and converted through the loop's frame rate
at construction, never held as raw frame counts. A frame count silently means a
different amount of real time on a device running at a different rate, which is
exactly how a duration tuned at one cadence became wrong at another in the yawn
work. The conversion happens once, here, so callers reason in wall clock time.

The track never reaches ALARM. Eyes off road is a lesser cue than sustained
drowsiness, matching how the distraction track is bounded at AUDIBLE.
"""

from __future__ import annotations

import math

from safeeyes.alert.state_machine import AlertTier


class EyesOffRoadTrack:
    def __init__(
        self,
        min_seconds: float = 2.0,
        fps: float = 10.0,
        audible_seconds: float | None = None,
    ) -> None:
        if min_seconds <= 0:
            raise ValueError(f"min_seconds must be positive, got {min_seconds}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if audible_seconds is not None and audible_seconds < min_seconds:
            raise ValueError(
                f"audible_seconds {audible_seconds} must not precede min_seconds {min_seconds}"
            )

        self.min_seconds = min_seconds
        self.audible_seconds = audible_seconds
        self.fps = fps
        self._visual_steps = self._to_steps(min_seconds, fps)
        self._audible_steps = (
            self._to_steps(audible_seconds, fps) if audible_seconds is not None else None
        )
        self.reset()

    @staticmethod
    def _to_steps(seconds: float, fps: float) -> int:
        """Frames needed to cover a duration, rounded up so the bar is never short."""
        return max(1, math.ceil(seconds * fps))

    def reset(self) -> None:
        self._off_road_steps = 0

    def update(self, off_road: bool) -> AlertTier:
        if not off_road:
            self._off_road_steps = 0
            return AlertTier.NONE

        self._off_road_steps += 1
        if self._audible_steps is not None and self._off_road_steps >= self._audible_steps:
            return AlertTier.AUDIBLE
        if self._off_road_steps >= self._visual_steps:
            return AlertTier.VISUAL
        return AlertTier.NONE
