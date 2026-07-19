import pytest

from safeeyes.alert.replay import TierEvent
from safeeyes.alert.state_machine import AlertTier
from safeeyes.alert.validation import (
    DEFAULT_PARAMS,
    SWEEP_GRID,
    ClipReplay,
    count_alarms,
    first_alarm_frame,
    select_parameters,
    summarize_replays,
    sweep_parameters,
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


def test_default_params_match_live_loop_and_are_in_grid():
    assert DEFAULT_PARAMS == (5, 15, 45)
    assert DEFAULT_PARAMS in SWEEP_GRID


def test_sweep_parameters_runs_grid_over_cached_levels():
    sequences = [
        ("a", "alert", [0] * 50),
        ("d", "drowsy", [2] * 50),
    ]
    grid = [(2, 3, 100), (5, 15, 45)]
    rows = sweep_parameters(sequences, grid, AlertTier.AUDIBLE, fps=10.0)
    assert [r["params"] for r in rows] == [[2, 3, 100], [5, 15, 45]]
    assert rows[0]["drowsy_detection_rate"] == 1.0
    assert rows[0]["false_alarms_per_hour"] == 0.0


def _row(params, detection, fa_per_hour, median):
    return {
        "params": list(params),
        "drowsy_detection_rate": detection,
        "false_alarms_per_hour": fa_per_hour,
        "median_time_to_first_alert_s": median,
    }


def test_select_parameters_prefers_lowest_far_meeting_baseline_detection():
    rows = [
        _row((5, 15, 45), 0.8, 4.0, 30.0),
        _row((8, 15, 45), 0.8, 2.0, 40.0),
        _row((3, 8, 15), 0.9, 6.0, 10.0),
        _row((12, 40, 90), 0.7, 0.5, 60.0),
    ]
    assert select_parameters(rows)["params"] == [8, 15, 45]


def test_select_parameters_tie_breaks_on_faster_first_alert():
    rows = [
        _row((5, 15, 45), 0.8, 2.0, 30.0),
        _row((8, 25, 45), 0.8, 2.0, 20.0),
    ]
    assert select_parameters(rows)["params"] == [8, 25, 45]


def test_select_parameters_requires_baseline_row():
    with pytest.raises(ValueError):
        select_parameters([_row((3, 8, 15), 0.9, 1.0, 5.0)])


def test_select_parameters_rejects_baseline_without_drowsy_clips():
    rows = [_row(DEFAULT_PARAMS, None, 1.0, None)]
    with pytest.raises(ValueError):
        select_parameters(rows)
