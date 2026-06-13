import pytest

from safeeyes.alert.state_machine import AlertStateMachine, AlertTier

ALERT, LOW, DROWSY = 0, 1, 2


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
