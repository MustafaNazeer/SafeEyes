import pytest

from safeeyes.alert.gaze_track import EyesOffRoadTrack
from safeeyes.alert.state_machine import AlertTier, fuse_tiers


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _hold(track, clock, off_road: bool, seconds: float, fps: float) -> AlertTier:
    """Feed the track for a span of wall clock time at a given frame rate."""
    tier = AlertTier.NONE
    step = 1.0 / fps
    elapsed = 0.0
    while elapsed < seconds:
        clock.advance(step)
        elapsed += step
        tier = track.update(off_road)
    return tier


def _build(min_seconds=1.0, audible_seconds=None):
    clock = FakeClock()
    return EyesOffRoadTrack(
        min_seconds=min_seconds, audible_seconds=audible_seconds, clock=clock
    ), clock


def test_a_brief_glance_does_not_fire():
    """Checking a mirror must not raise an alert."""
    track, clock = _build(min_seconds=1.0)
    assert _hold(track, clock, True, 0.5, fps=10.0) == AlertTier.NONE


def test_sustained_off_road_fires():
    track, clock = _build(min_seconds=1.0)
    assert _hold(track, clock, True, 1.2, fps=10.0) == AlertTier.VISUAL


def test_looking_back_at_the_road_stands_the_alert_down():
    track, clock = _build(min_seconds=1.0)
    _hold(track, clock, True, 1.2, fps=10.0)
    assert track.update(False) == AlertTier.NONE


def test_the_same_wall_clock_duration_holds_at_any_frame_rate():
    """The bar is one second of real time however many frames arrive in it.

    The track is told nothing about the loop's rate. This is the defect the
    live loop hit: it was constructed with a nominal 11 fps while the loop
    actually ran at 7, which silently stretched a 2 second bar to 3.1 seconds.
    """
    for fps in (5.0, 11.0, 30.0):
        track, clock = _build(min_seconds=1.0)
        assert _hold(track, clock, True, 0.9, fps=fps) == AlertTier.NONE
        assert _hold(track, clock, True, 0.2, fps=fps) == AlertTier.VISUAL


def test_a_loop_running_slower_than_expected_still_fires_on_time():
    """The regression this fix exists for.

    At 7 fps a two second bar must still be two seconds of wall clock, not the
    3.1 seconds a frame count derived from a nominal 11 fps would have needed.
    """
    track, clock = _build(min_seconds=2.0)
    assert _hold(track, clock, True, 1.9, fps=7.0) == AlertTier.NONE
    assert _hold(track, clock, True, 0.2, fps=7.0) == AlertTier.VISUAL


def test_a_longer_gaze_escalates_to_audible():
    track, clock = _build(min_seconds=1.0, audible_seconds=2.0)
    assert _hold(track, clock, True, 1.2, fps=10.0) == AlertTier.VISUAL
    assert _hold(track, clock, True, 1.0, fps=10.0) == AlertTier.AUDIBLE


def test_the_track_never_reaches_alarm():
    """Eyes off road is a lesser cue than sustained drowsiness."""
    track, clock = _build(min_seconds=0.5, audible_seconds=1.0)
    assert _hold(track, clock, True, 50.0, fps=10.0) == AlertTier.AUDIBLE


def test_interrupted_glances_do_not_accumulate():
    track, clock = _build(min_seconds=1.0)
    for _ in range(6):
        _hold(track, clock, True, 0.8, fps=10.0)
        track.update(False)
    assert _hold(track, clock, True, 0.8, fps=10.0) == AlertTier.NONE


def test_reset_clears_the_state():
    track, clock = _build(min_seconds=1.0)
    _hold(track, clock, True, 2.0, fps=10.0)
    track.reset()
    assert _hold(track, clock, True, 0.5, fps=10.0) == AlertTier.NONE


def test_it_fuses_with_the_other_tracks_by_severity():
    track, clock = _build(min_seconds=1.0)
    gaze = _hold(track, clock, True, 1.2, fps=10.0)
    assert fuse_tiers(AlertTier.NONE, gaze) == AlertTier.VISUAL
    assert fuse_tiers(AlertTier.ALARM, gaze) == AlertTier.ALARM


def test_a_non_positive_duration_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        EyesOffRoadTrack(min_seconds=0.0)


def test_audible_must_not_precede_the_visual_threshold():
    with pytest.raises(ValueError, match="audible"):
        EyesOffRoadTrack(min_seconds=2.0, audible_seconds=1.0)
