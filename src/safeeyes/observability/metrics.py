"""Rolling per-window run metrics for the live loop.

Accumulates per-frame processing latency and face-detection counts, then reports
frames, throughput, median and 95th percentile latency, and the face-detection
rate for the window. Throughput is real wall-clock frames per second measured
against an injectable clock, so the arithmetic is tested deterministically; in
normal use it is the process performance counter. A window is emitted and reset
on a fixed interval by the observer that owns it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricsSummary:
    frames: int
    fps: float
    lat_ms_p50: float
    lat_ms_p95: float
    face_rate: float


class RunMetrics:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._latencies_s: list[float] = []
        self._faces = 0
        self._window_start = clock()

    def record(self, latency_s: float, face_detected: bool) -> None:
        self._latencies_s.append(latency_s)
        if face_detected:
            self._faces += 1

    def summary(self) -> MetricsSummary:
        frames = len(self._latencies_s)
        if frames == 0:
            return MetricsSummary(0, 0.0, 0.0, 0.0, 0.0)
        elapsed = self._clock() - self._window_start
        latencies_ms = np.asarray(self._latencies_s, dtype=np.float64) * 1000.0
        return MetricsSummary(
            frames=frames,
            fps=float(frames / elapsed) if elapsed > 0.0 else 0.0,
            lat_ms_p50=float(np.percentile(latencies_ms, 50)),
            lat_ms_p95=float(np.percentile(latencies_ms, 95)),
            face_rate=self._faces / frames,
        )

    def reset(self) -> None:
        self._latencies_s = []
        self._faces = 0
        self._window_start = self._clock()
