from itertools import count

import numpy as np
import pytest

from safeeyes.edge.benchmark import BenchmarkResult, benchmark_model
from safeeyes.edge.export import export_temporal_onnx
from safeeyes.edge.runtime import OnnxModel
from safeeyes.temporal.model import TemporalGRU


def _model(tmp_path):
    torch_model = TemporalGRU(n_features=5, num_classes=3).eval()
    path = export_temporal_onnx(torch_model, tmp_path / "t.onnx", n_features=5)
    return OnnxModel(path), np.zeros((1, 150, 5), dtype=np.float32)


def test_benchmark_latency_math_from_injected_clock(tmp_path) -> None:
    model, sample = _model(tmp_path)
    ticks = count(0.0, 0.002)
    result = benchmark_model(model, sample, runs=5, warmup=0, clock=lambda: next(ticks))
    assert isinstance(result, BenchmarkResult)
    assert result.runs == 5
    assert result.mean_ms == pytest.approx(2.0)
    assert result.p50_ms == pytest.approx(2.0)
    assert result.fps == pytest.approx(500.0)


def test_benchmark_real_clock_invariants(tmp_path) -> None:
    model, sample = _model(tmp_path)
    result = benchmark_model(model, sample, runs=20, warmup=5)
    assert result.runs == 20
    assert result.mean_ms > 0.0
    assert result.p95_ms >= result.p50_ms
    assert result.fps > 0.0


def test_benchmark_warmup_not_counted(tmp_path) -> None:
    model, sample = _model(tmp_path)
    calls = {"n": 0}

    def clock() -> float:
        calls["n"] += 1
        return calls["n"] * 0.001

    benchmark_model(model, sample, runs=4, warmup=3, clock=clock)
    assert calls["n"] == 8  # two clock reads per timed run, warmup untimed
