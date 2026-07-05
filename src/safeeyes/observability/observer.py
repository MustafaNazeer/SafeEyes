"""Run observer: turns the live loop into a stream of structured telemetry.

Given a per-frame call, it records latency and face-detection into the rolling
metrics, emits an event whenever the alert tier changes or the face is lost or
regained, and flushes a metrics summary every fixed interval. The camera loop
calls ``start`` once, ``observe`` per frame, and ``stop`` at shutdown. Timing and
output are injected, so every decision here is unit tested without a camera.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from safeeyes.observability.emitter import JsonlEmitter
from safeeyes.observability.metrics import RunMetrics


class RunObserver:
    def __init__(
        self,
        emitter: JsonlEmitter,
        metrics: RunMetrics,
        metrics_interval_s: float = 5.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._emitter = emitter
        self._metrics = metrics
        self._interval = metrics_interval_s
        self._clock = clock
        self._last_tier: str | None = None
        self._last_face: bool | None = None
        self._last_flush = clock()

    def start(self, config: Mapping[str, object]) -> None:
        self._emitter.emit("start", config=dict(config))
        self._last_flush = self._clock()

    def observe(
        self,
        *,
        tier: str,
        fatigue: int,
        face_detected: bool,
        latency_s: float,
    ) -> None:
        self._metrics.record(latency_s, face_detected=face_detected)

        if self._last_tier is not None and tier != self._last_tier:
            self._emitter.emit(
                "tier_change", **{"from": self._last_tier, "to": tier, "fatigue": fatigue}
            )
        self._last_tier = tier

        if self._last_face is not None and face_detected != self._last_face:
            self._emitter.emit("face_regained" if face_detected else "face_lost")
        self._last_face = face_detected

        if self._clock() - self._last_flush >= self._interval:
            self._flush()

    def stop(self) -> None:
        self._flush()
        self._emitter.emit("stop")

    def _flush(self) -> None:
        summary = self._metrics.summary()
        self._emitter.emit(
            "metrics",
            frames=summary.frames,
            fps=summary.fps,
            lat_ms_p50=summary.lat_ms_p50,
            lat_ms_p95=summary.lat_ms_p95,
            face_rate=summary.face_rate,
        )
        self._metrics.reset()
        self._last_flush = self._clock()
