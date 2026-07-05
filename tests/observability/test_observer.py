import io
import json

import pytest

from safeeyes.observability.emitter import JsonlEmitter
from safeeyes.observability.metrics import RunMetrics
from safeeyes.observability.observer import RunObserver


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _build(interval: float = 1_000.0):
    stream = io.StringIO()
    clock = FakeClock()
    emitter = JsonlEmitter(stream=stream, clock=lambda: 0.0)
    metrics = RunMetrics(clock=clock)
    observer = RunObserver(emitter, metrics, metrics_interval_s=interval, clock=clock)
    return observer, stream, clock


def _records(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_start_emits_the_run_config() -> None:
    observer, stream, _ = _build()

    observer.start({"window": 150, "backend": "torch"})

    records = _records(stream)
    assert len(records) == 1
    assert records[0]["event"] == "start"
    assert records[0]["config"] == {"window": 150, "backend": "torch"}


def test_tier_change_emitted_only_on_a_transition() -> None:
    observer, stream, _ = _build()
    observer.start({})

    observer.observe(tier="none", fatigue=0, face_detected=True, latency_s=0.01)
    observer.observe(tier="none", fatigue=0, face_detected=True, latency_s=0.01)
    observer.observe(tier="visual", fatigue=1, face_detected=True, latency_s=0.01)

    events = [r["event"] for r in _records(stream)]
    assert events == ["start", "tier_change"]
    change = next(r for r in _records(stream) if r["event"] == "tier_change")
    assert change["from"] == "none"
    assert change["to"] == "visual"
    assert change["fatigue"] == 1


def test_face_lost_and_regained_are_edge_triggered() -> None:
    observer, stream, _ = _build()
    observer.start({})

    for face in [True, True, False, False, True]:
        observer.observe(tier="none", fatigue=0, face_detected=face, latency_s=0.01)

    events = [r["event"] for r in _records(stream) if r["event"].startswith("face_")]
    assert events == ["face_lost", "face_regained"]


def test_metrics_flushed_once_the_interval_elapses() -> None:
    observer, stream, clock = _build(interval=5.0)
    observer.start({})

    for _ in range(3):
        observer.observe(tier="none", fatigue=0, face_detected=True, latency_s=0.02)
    assert not any(r["event"] == "metrics" for r in _records(stream))

    clock.t = 5.0
    observer.observe(tier="none", fatigue=0, face_detected=True, latency_s=0.02)

    metrics = [r for r in _records(stream) if r["event"] == "metrics"]
    assert len(metrics) == 1
    assert metrics[0]["frames"] == 4
    assert metrics[0]["fps"] == pytest.approx(0.8)  # 4 frames over 5 s


def test_stop_flushes_a_final_metrics_line_then_a_stop() -> None:
    observer, stream, clock = _build(interval=5.0)
    observer.start({})
    observer.observe(tier="none", fatigue=0, face_detected=True, latency_s=0.02)

    clock.t = 1.0
    observer.stop()

    events = [r["event"] for r in _records(stream)]
    assert events[-2:] == ["metrics", "stop"]
