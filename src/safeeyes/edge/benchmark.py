"""Measure inference latency and throughput for an ONNX model.

The harness warms up the session, then times a fixed number of single-sample
inferences and reports mean, median, and 95th percentile latency in
milliseconds plus the throughput implied by the mean. The clock is injectable so
the latency arithmetic is tested deterministically; in normal use it is the
process performance counter. The headline Raspberry Pi numbers come from running
this against the quantized model on the Pi itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from safeeyes.edge.runtime import OnnxModel


@dataclass(frozen=True)
class BenchmarkResult:
    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float


def benchmark_model(
    model: OnnxModel,
    sample: np.ndarray,
    runs: int = 100,
    warmup: int = 10,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    if runs < 1:
        raise ValueError("runs must be at least 1")

    for _ in range(warmup):
        model.run(sample)

    latencies_ms = np.empty(runs, dtype=np.float64)
    for i in range(runs):
        start = clock()
        model.run(sample)
        end = clock()
        latencies_ms[i] = (end - start) * 1000.0

    mean_ms = float(latencies_ms.mean())
    return BenchmarkResult(
        runs=runs,
        mean_ms=mean_ms,
        p50_ms=float(np.percentile(latencies_ms, 50)),
        p95_ms=float(np.percentile(latencies_ms, 95)),
        fps=(1000.0 / mean_ms) if mean_ms > 0 else float("inf"),
    )
