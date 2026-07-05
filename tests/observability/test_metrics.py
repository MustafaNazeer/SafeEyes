import pytest

from safeeyes.observability.metrics import MetricsSummary, RunMetrics


class FakeClock:
    """A hand-advanced clock so window elapsed time is deterministic."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_summary_reports_percentiles_rate_and_fps() -> None:
    clock = FakeClock()
    metrics = RunMetrics(clock=clock)

    for latency_s, face in [(0.010, True), (0.020, True), (0.030, False), (0.040, True)]:
        metrics.record(latency_s, face_detected=face)

    clock.t = 2.0
    summary = metrics.summary()

    assert isinstance(summary, MetricsSummary)
    assert summary.frames == 4
    assert summary.lat_ms_p50 == pytest.approx(25.0)
    assert summary.lat_ms_p95 == pytest.approx(38.5)
    assert summary.face_rate == pytest.approx(0.75)
    assert summary.fps == pytest.approx(2.0)  # 4 frames over 2.0 s of wall time


def test_empty_window_summary_is_zeroed_without_error() -> None:
    clock = FakeClock()
    metrics = RunMetrics(clock=clock)

    clock.t = 5.0
    summary = metrics.summary()

    assert summary.frames == 0
    assert summary.fps == 0.0
    assert summary.lat_ms_p50 == 0.0
    assert summary.lat_ms_p95 == 0.0
    assert summary.face_rate == 0.0


def test_zero_elapsed_yields_zero_fps_not_division_error() -> None:
    clock = FakeClock()
    metrics = RunMetrics(clock=clock)

    metrics.record(0.01, face_detected=True)
    metrics.record(0.01, face_detected=True)
    summary = metrics.summary()  # no time has passed

    assert summary.frames == 2
    assert summary.fps == 0.0


def test_reset_starts_a_fresh_window() -> None:
    clock = FakeClock()
    metrics = RunMetrics(clock=clock)

    metrics.record(0.100, face_detected=False)
    clock.t = 10.0
    metrics.reset()

    metrics.record(0.020, face_detected=True)
    clock.t = 11.0
    summary = metrics.summary()

    assert summary.frames == 1
    assert summary.lat_ms_p50 == pytest.approx(20.0)
    assert summary.face_rate == pytest.approx(1.0)
    assert summary.fps == pytest.approx(1.0)  # 1 frame over the 1.0 s since reset
