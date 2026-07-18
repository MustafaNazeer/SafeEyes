import pytest

from safeeyes.alert.state_machine import (
    AlertStateMachine,
    AlertTier,
    DistractionAlertTrack,
    fuse_tiers,
)

ALERT, LOW, DROWSY = 0, 1, 2


def _feed_track(track: DistractionAlertTrack, distracted: bool, times: int) -> AlertTier:
    tier = track.current_tier
    for _ in range(times):
        tier = track.update(distracted)
    return tier


def test_distraction_escalates_only_after_sustained_signal() -> None:
    track = DistractionAlertTrack(escalate_steps=3, de_escalate_steps=8, audible_after=15)
    assert track.update(True) == AlertTier.NONE
    assert track.update(True) == AlertTier.NONE
    assert track.update(True) == AlertTier.VISUAL


def test_distraction_transient_signal_does_not_commit() -> None:
    track = DistractionAlertTrack(escalate_steps=3)
    track.update(True)
    track.update(True)
    assert track.update(False) == AlertTier.NONE
    # the streak reset, so a fresh pair is still not enough
    assert track.update(True) == AlertTier.NONE
    assert track.update(True) == AlertTier.NONE


def test_distraction_de_escalates_slowly() -> None:
    track = DistractionAlertTrack(escalate_steps=3, de_escalate_steps=4, audible_after=15)
    _feed_track(track, True, 3)
    assert track.current_tier == AlertTier.VISUAL
    assert _feed_track(track, False, 3) == AlertTier.VISUAL  # not yet cleared
    assert track.update(False) == AlertTier.NONE  # fourth false clears


def test_distraction_escalates_to_audible_when_sustained() -> None:
    # commit costs escalate_steps (2), then audible_after (5) committed steps are
    # counted from the commit; AUDIBLE therefore lands on the 6th sustained True.
    track = DistractionAlertTrack(escalate_steps=2, de_escalate_steps=8, audible_after=5)
    assert _feed_track(track, True, 5) == AlertTier.VISUAL
    assert track.update(True) == AlertTier.AUDIBLE


def test_fuse_tiers_returns_the_max() -> None:
    assert fuse_tiers(AlertTier.NONE, AlertTier.VISUAL) == AlertTier.VISUAL
    assert fuse_tiers(AlertTier.ALARM, AlertTier.VISUAL) == AlertTier.ALARM
    assert fuse_tiers(AlertTier.NONE, AlertTier.NONE) == AlertTier.NONE
    assert fuse_tiers() == AlertTier.NONE
    assert isinstance(fuse_tiers(AlertTier.AUDIBLE), AlertTier)


def test_distraction_track_rejects_bad_thresholds() -> None:
    with pytest.raises(ValueError):
        DistractionAlertTrack(escalate_steps=0)
    with pytest.raises(ValueError):
        DistractionAlertTrack(audible_after=0)


def _feed(machine: AlertStateMachine, level: int, times: int) -> AlertTier:
    tier = machine.current_tier
    for _ in range(times):
        tier = machine.update(level)
    return tier


def test_starts_at_no_alert() -> None:
    assert AlertStateMachine().current_tier == AlertTier.NONE


def test_single_drowsy_frame_does_not_fire() -> None:
    m = AlertStateMachine(escalate_steps=3)
    assert m.update(DROWSY) == AlertTier.NONE


def test_transient_drowsy_then_alert_does_not_escalate() -> None:
    m = AlertStateMachine(escalate_steps=3, de_escalate_steps=4)
    m.update(DROWSY)
    m.update(DROWSY)  # 2 < 3, not committed
    assert m.update(ALERT) == AlertTier.NONE


def test_sustained_drowsy_reaches_audible() -> None:
    m = AlertStateMachine(escalate_steps=3, alarm_after=10)
    assert _feed(m, DROWSY, 3) == AlertTier.AUDIBLE


def test_prolonged_drowsy_escalates_to_alarm() -> None:
    m = AlertStateMachine(escalate_steps=3, alarm_after=5)
    _feed(m, DROWSY, 3)  # committed drowsy -> audible
    assert _feed(m, DROWSY, 5) == AlertTier.ALARM


def test_sustained_low_vigilance_reaches_visual() -> None:
    m = AlertStateMachine(escalate_steps=3)
    assert _feed(m, LOW, 3) == AlertTier.VISUAL


def test_hysteresis_recovery_is_sticky() -> None:
    m = AlertStateMachine(escalate_steps=3, de_escalate_steps=5, alarm_after=100)
    _feed(m, DROWSY, 3)  # committed drowsy -> audible
    # a few alert frames below the de-escalation threshold must not drop the tier
    assert _feed(m, ALERT, 4) == AlertTier.AUDIBLE
    # crossing the threshold finally de-escalates toward no alert
    assert _feed(m, ALERT, 1) == AlertTier.NONE


def test_invalid_level_raises() -> None:
    with pytest.raises(ValueError):
        AlertStateMachine().update(7)


def test_reset_returns_to_no_alert() -> None:
    m = AlertStateMachine(escalate_steps=3)
    _feed(m, DROWSY, 3)
    m.reset()
    assert m.current_tier == AlertTier.NONE
