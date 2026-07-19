import pytest

from safeeyes.alert.replay import TierEvent
from safeeyes.alert.state_machine import AlertTier
from safeeyes.alert.validation import (
    ClipReplay,
    count_alarms,
    first_alarm_frame,
    summarize_replays,
)


def _events(*pairs):
    return [TierEvent(frame=f, tier=t) for f, t in pairs]


def test_count_alarms_counts_rising_edges_only():
    events = _events(
        (10, AlertTier.VISUAL),
        (20, AlertTier.AUDIBLE),
        (30, AlertTier.ALARM),
        (40, AlertTier.NONE),
        (50, AlertTier.AUDIBLE),
    )
    assert count_alarms(events, AlertTier.AUDIBLE) == 2
    assert count_alarms(events, AlertTier.ALARM) == 1
    assert count_alarms(events, AlertTier.VISUAL) == 2
    assert count_alarms([], AlertTier.AUDIBLE) == 0


def test_first_alarm_frame():
    events = _events((10, AlertTier.VISUAL), (25, AlertTier.AUDIBLE))
    assert first_alarm_frame(events, AlertTier.AUDIBLE) == 25
    assert first_alarm_frame(events, AlertTier.ALARM) is None


def test_summarize_replays_headline_metrics():
    clips = [
        ClipReplay("a", "alert", 36000, _events((100, AlertTier.AUDIBLE))),
        ClipReplay("b", "low_vigilance", 36000, []),
        ClipReplay("c", "drowsy", 36000, _events((200, AlertTier.AUDIBLE))),
        ClipReplay("d", "drowsy", 36000, []),
    ]
    s = summarize_replays(clips, AlertTier.AUDIBLE, fps=10.0)
    assert s["n_clips"] == 4
    assert s["n_not_drowsy_clips"] == 2
    assert s["n_drowsy_clips"] == 2
    assert s["false_alarms_per_hour"] == pytest.approx(0.5)
    assert s["fraction_not_drowsy_clips_with_alarm"] == pytest.approx(0.5)
    assert s["false_alarms_per_hour_alert_clips_only"] == pytest.approx(1.0)
    assert s["drowsy_detection_rate"] == pytest.approx(0.5)
    assert s["median_time_to_first_alert_s"] == pytest.approx(20.0)


def test_summarize_replays_no_drowsy_detections_yields_none_median():
    clips = [ClipReplay("d", "drowsy", 1000, [])]
    s = summarize_replays(clips, AlertTier.AUDIBLE, fps=10.0)
    assert s["drowsy_detection_rate"] == 0.0
    assert s["median_time_to_first_alert_s"] is None


def test_summarize_replays_rejects_unknown_label():
    with pytest.raises(ValueError):
        summarize_replays([ClipReplay("x", "sleepy", 10, [])], AlertTier.AUDIBLE, fps=10.0)
