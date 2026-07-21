import pytest

from safeeyes.alert.gaze_track import EyesOffRoadTrack
from safeeyes.alert.state_machine import AlertTier, fuse_tiers


def _run(track, off_road: bool, steps: int) -> AlertTier:
    tier = AlertTier.NONE
    for _ in range(steps):
        tier = track.update(off_road)
    return tier


def test_a_brief_glance_does_not_fire():
    """Checking a mirror must not raise an alert."""
    track = EyesOffRoadTrack(min_seconds=1.0, fps=10.0)
    assert _run(track, True, 5) == AlertTier.NONE


def test_sustained_off_road_fires():
    track = EyesOffRoadTrack(min_seconds=1.0, fps=10.0)
    assert _run(track, True, 12) == AlertTier.VISUAL


def test_looking_back_at_the_road_stands_the_alert_down():
    track = EyesOffRoadTrack(min_seconds=1.0, fps=10.0)
    _run(track, True, 12)
    assert _run(track, False, 1) == AlertTier.NONE


def test_the_same_wall_clock_duration_holds_at_a_different_frame_rate():
    """The duration is seconds, not frames.

    A track built for 1 second must fire after 1 second of real time whether
    the loop runs at 10 or 30 fps. Storing a raw frame count here is the bug
    that the yawn work walked into when a duration tuned at one cadence was
    carried to another.
    """
    slow = EyesOffRoadTrack(min_seconds=1.0, fps=10.0)
    fast = EyesOffRoadTrack(min_seconds=1.0, fps=30.0)

    assert _run(slow, True, 9) == AlertTier.NONE
    assert _run(fast, True, 29) == AlertTier.NONE
    assert slow.update(True) == AlertTier.VISUAL
    assert fast.update(True) == AlertTier.VISUAL


def test_a_longer_gaze_escalates_to_audible():
    track = EyesOffRoadTrack(min_seconds=1.0, audible_seconds=2.0, fps=10.0)
    assert _run(track, True, 12) == AlertTier.VISUAL
    assert _run(track, True, 10) == AlertTier.AUDIBLE


def test_the_track_never_reaches_alarm():
    """Eyes off road is a lesser cue than sustained drowsiness."""
    track = EyesOffRoadTrack(min_seconds=0.5, audible_seconds=1.0, fps=10.0)
    assert _run(track, True, 500) == AlertTier.AUDIBLE


def test_interrupted_glances_do_not_accumulate():
    track = EyesOffRoadTrack(min_seconds=1.0, fps=10.0)
    for _ in range(6):
        _run(track, True, 8)
        track.update(False)
    assert _run(track, True, 8) == AlertTier.NONE


def test_reset_clears_the_state():
    track = EyesOffRoadTrack(min_seconds=1.0, fps=10.0)
    _run(track, True, 20)
    track.reset()
    assert _run(track, True, 5) == AlertTier.NONE


def test_it_fuses_with_the_other_tracks_by_severity():
    track = EyesOffRoadTrack(min_seconds=1.0, fps=10.0)
    gaze = _run(track, True, 12)
    assert fuse_tiers(AlertTier.NONE, gaze) == AlertTier.VISUAL
    assert fuse_tiers(AlertTier.ALARM, gaze) == AlertTier.ALARM


def test_a_non_positive_duration_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        EyesOffRoadTrack(min_seconds=0.0, fps=10.0)


def test_a_non_positive_frame_rate_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        EyesOffRoadTrack(min_seconds=1.0, fps=0.0)


def test_audible_must_not_precede_the_visual_threshold():
    with pytest.raises(ValueError, match="audible"):
        EyesOffRoadTrack(min_seconds=2.0, audible_seconds=1.0, fps=10.0)
