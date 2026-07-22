import pytest

from safeeyes.alert.gaze_smoothing import GazeZoneSmoother


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _feed(smoother, clock, zones, fps=10.0):
    """Feed one zone per frame at a given rate, returning the last smoothed value."""
    out = None
    for zone in zones:
        clock.advance(1.0 / fps)
        out = smoother.update(zone)
    return out


def _build(window_seconds=1.0):
    clock = FakeClock()
    return GazeZoneSmoother(window_seconds=window_seconds, clock=clock), clock


def test_a_steady_zone_passes_through():
    smoother, clock = _build()
    assert _feed(smoother, clock, ["front"] * 10) == "front"


def test_a_single_stray_frame_does_not_flip_the_output():
    """The defect this exists for.

    The raw classifier flips one to two times per second on a steady gaze, and
    every flip reached the alert track unfiltered.
    """
    smoother, clock = _build()
    _feed(smoother, clock, ["front"] * 9)
    assert smoother.update("infotainment") == "front"


def test_a_sustained_change_does_take_effect():
    smoother, clock = _build()
    _feed(smoother, clock, ["front"] * 10)
    assert _feed(smoother, clock, ["infotainment"] * 10) == "infotainment"


def test_observations_older_than_the_window_are_forgotten():
    smoother, clock = _build(window_seconds=1.0)
    _feed(smoother, clock, ["front"] * 10)
    clock.advance(5.0)
    assert smoother.update("infotainment") == "infotainment"


def test_the_window_is_wall_clock_so_the_frame_rate_does_not_matter():
    """Same defect class as the eyes off road duration bug: seconds, not frames."""
    for fps in (7.0, 30.0):
        smoother, clock = _build(window_seconds=1.0)
        # Three quarters of a second of front, then a quarter second of noise.
        _feed(smoother, clock, ["front"] * int(0.75 * fps), fps=fps)
        assert _feed(smoother, clock, ["infotainment"] * int(0.25 * fps), fps=fps) == "front"


def test_no_face_yields_no_zone():
    smoother, clock = _build()
    _feed(smoother, clock, ["front"] * 10)
    assert smoother.update(None) is None


def test_a_lost_face_clears_the_history_rather_than_holding_a_stale_zone():
    smoother, clock = _build()
    _feed(smoother, clock, ["front"] * 10)
    smoother.update(None)
    assert smoother.update("infotainment") == "infotainment"


def test_before_any_observation_there_is_no_zone():
    smoother, _ = _build()
    assert smoother.zone is None


def test_a_tie_resolves_to_the_most_recent_zone():
    smoother, clock = _build(window_seconds=1.0)
    _feed(smoother, clock, ["front", "infotainment"] * 2, fps=10.0)
    assert smoother.update("infotainment") == "infotainment"


def test_reset_clears_the_history():
    smoother, clock = _build()
    _feed(smoother, clock, ["front"] * 10)
    smoother.reset()
    assert smoother.update("infotainment") == "infotainment"


def test_a_non_positive_window_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        GazeZoneSmoother(window_seconds=0.0)
