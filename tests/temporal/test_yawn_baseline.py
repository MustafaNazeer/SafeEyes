import numpy as np

from safeeyes.temporal.yawn_baseline import precision_recall, sweep_min_duration

LONG = np.array([0.1] + [0.9] * 10 + [0.1])
SHORT = np.array([0.1, 0.9, 0.1])


def test_precision_recall_counts_video_level():
    videos = [("a", "Yawning", LONG), ("b", "Talking", SHORT)]
    precision, recall, tp, fp, fn = precision_recall(videos, 0.5, min_duration=5)
    assert (tp, fp, fn) == (1, 0, 0)
    assert precision == 1.0
    assert recall == 1.0


def test_short_events_become_false_negatives_when_duration_is_too_strict():
    videos = [("a", "Yawning", SHORT)]
    precision, recall, tp, fp, fn = precision_recall(videos, 0.5, min_duration=5)
    assert (tp, fp, fn) == (0, 0, 1)
    assert recall == 0.0


def test_sweep_prefers_highest_precision_above_the_recall_floor():
    videos = [("a", "Yawning", LONG), ("b", "Talking", SHORT), ("c", "Normal", SHORT)]
    result = sweep_min_duration(videos, 0.5, durations=range(1, 8), min_recall=0.9)
    assert result["selected"] == 2
    assert result["floor_met"] is True


def test_sweep_breaks_ties_toward_the_smaller_duration():
    videos = [("a", "Yawning", LONG), ("b", "Normal", SHORT)]
    result = sweep_min_duration(videos, 0.5, durations=range(1, 6), min_recall=0.9)
    assert result["selected"] == 2


def test_sweep_reports_when_no_duration_meets_the_floor():
    videos = [("a", "Yawning", SHORT), ("b", "Yawning", SHORT)]
    result = sweep_min_duration(videos, 0.5, durations=range(4, 8), min_recall=0.9)
    assert result["floor_met"] is False
