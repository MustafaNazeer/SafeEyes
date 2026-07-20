import numpy as np
import pytest

from safeeyes.temporal.yawn_validation import (
    MAR_YAWN_THRESHOLD,
    derive_mar_threshold,
    event_runs,
    score_videos,
    threshold_curve,
    video_predicts_yawning,
)


def test_derive_mar_threshold_is_the_requested_percentile():
    mar = np.arange(1000, dtype=float) / 1000.0
    assert derive_mar_threshold(mar, percentile=99.0) == pytest.approx(0.98901, abs=1e-4)


def test_derive_mar_threshold_rejects_empty():
    with pytest.raises(ValueError):
        derive_mar_threshold(np.array([]))


def test_registered_threshold_is_pinned():
    assert MAR_YAWN_THRESHOLD == pytest.approx(0.616703, abs=1e-6)


def _mar_with_onsets(n_onsets, threshold=0.5):
    trace = [threshold - 0.2] * 5
    for _ in range(n_onsets):
        trace += [threshold + 0.2] * 3 + [threshold - 0.2] * 3
    return np.array(trace)


def test_video_predicts_yawning_requires_an_onset():
    assert video_predicts_yawning(_mar_with_onsets(1), 0.5) is True
    assert video_predicts_yawning(_mar_with_onsets(0), 0.5) is False
    assert video_predicts_yawning(np.array([]), 0.5) is False


def test_score_videos_precision_recall_and_talking_breakout():
    videos = [
        ("Yawning", _mar_with_onsets(2)),
        ("Talking&Yawning", _mar_with_onsets(1)),
        ("Normal", _mar_with_onsets(0)),
        ("Talking", _mar_with_onsets(1)),
    ]
    s = score_videos(videos, 0.5)
    assert s["n_videos"] == 4
    assert s["n_yawning_true"] == 2
    assert s["recall"] == 1.0
    assert s["precision"] == 2 / 3
    assert s["talking_false_positive_rate"] == 1.0
    assert s["per_category"]["Normal"] == {"n": 1, "predicted_yawning": 0}


def test_threshold_curve_monotone_recall():
    videos = [("Yawning", _mar_with_onsets(1)), ("Normal", _mar_with_onsets(0))]
    curve = threshold_curve(videos, [0.4, 0.9])
    assert curve[0]["threshold"] == 0.4
    assert curve[0]["recall"] == 1.0
    assert curve[1]["recall"] == 0.0


def test_event_runs_returns_inclusive_bounds():
    mar = np.array([0.1, 0.9, 0.9, 0.1, 0.9])
    assert event_runs(mar, 0.5) == [(1, 2), (4, 4)]


def test_event_runs_handles_a_run_touching_both_ends():
    mar = np.array([0.9, 0.9, 0.9])
    assert event_runs(mar, 0.5) == [(0, 2)]


def test_event_runs_on_empty_input():
    assert event_runs(np.array([]), 0.5) == []


def test_min_duration_rejects_a_short_excursion():
    mar = np.array([0.1, 0.9, 0.9, 0.1])
    assert video_predicts_yawning(mar, 0.5, min_duration=2) is True
    assert video_predicts_yawning(mar, 0.5, min_duration=3) is False


def test_min_duration_boundary_is_inclusive():
    mar = np.array([0.1, 0.9, 0.9, 0.9, 0.1])
    assert video_predicts_yawning(mar, 0.5, min_duration=3) is True


def test_default_min_duration_preserves_phase_12_behavior():
    mar = np.array([0.1, 0.9, 0.1])
    assert video_predicts_yawning(mar, 0.5) is True
