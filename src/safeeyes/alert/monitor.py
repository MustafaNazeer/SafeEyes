"""Driver monitor pipeline: fuse the fatigue and distraction tracks.

Wraps the existing drowsiness pipeline and the distraction track (scheduler plus
its debounced alert track) into one per frame step, so the runtime loop drives a
single object. The two tracks are independent: fatigue comes from the geometric
feature window, distraction from the raw frame on its Nth-frame cadence. The
overall tier is the max of the two, so neither cue masks the other. This module
is torch free so it runs inside the edge loop on the Pi.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from safeeyes.alert.state_machine import AlertTier, DistractionAlertTrack, fuse_tiers
from safeeyes.distraction.scheduler import DistractionState


class _Drowsiness(Protocol):
    def process(self, features: Sequence[float] | np.ndarray) -> AlertTier: ...

    @property
    def current_tier(self) -> AlertTier: ...

    @property
    def fatigue_level(self) -> int: ...


class _Scheduler(Protocol):
    def update(self, frame: np.ndarray | None) -> DistractionState: ...


@dataclass(frozen=True)
class MonitorState:
    tier: AlertTier
    fatigue_tier: AlertTier
    distraction_tier: AlertTier
    fatigue_level: int
    distraction: DistractionState


class DriverMonitorPipeline:
    def __init__(
        self,
        drowsiness: _Drowsiness,
        scheduler: _Scheduler,
        distraction_track: DistractionAlertTrack,
    ) -> None:
        self._drowsiness = drowsiness
        self._scheduler = scheduler
        self._track = distraction_track

    def process(
        self,
        features: Sequence[float] | np.ndarray | None,
        frame: np.ndarray | None,
    ) -> MonitorState:
        # Fatigue only advances on frames with a detected face (features present);
        # distraction advances every frame, since a distracted driver may have
        # turned away from the camera.
        if features is not None:
            fatigue_tier = self._drowsiness.process(features)
        else:
            fatigue_tier = self._drowsiness.current_tier
        distraction = self._scheduler.update(frame)
        distraction_tier = self._track.update(distraction.distracted)
        tier = fuse_tiers(fatigue_tier, distraction_tier)
        return MonitorState(
            tier=tier,
            fatigue_tier=fatigue_tier,
            distraction_tier=distraction_tier,
            fatigue_level=self._drowsiness.fatigue_level,
            distraction=distraction,
        )
