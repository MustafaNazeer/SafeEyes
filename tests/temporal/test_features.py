import pytest

from safeeyes.temporal.features import (
    count_onsets,
    event_rate_per_minute,
    mean_closure_duration,
    summarize_window,
)


def test_count_onsets_below_counts_falling_edges() -> None:
    # two separate closures
    ear = [0.30, 0.30, 0.05, 0.05, 0.30, 0.05]
    assert count_onsets(ear, threshold=0.1, mode="below") == 2


def test_count_onsets_counts_state_present_at_start() -> None:
    ear = [0.05, 0.05, 0.30]
    assert count_onsets(ear, threshold=0.1, mode="below") == 1


def test_count_onsets_above_counts_rising_edges() -> None:
    mar = [0.1, 0.6, 0.6, 0.1, 0.7]
    assert count_onsets(mar, threshold=0.5, mode="above") == 2


def test_count_onsets_all_out_of_state_is_zero() -> None:
    assert count_onsets([0.3, 0.3, 0.3], threshold=0.1, mode="below") == 0


def test_count_onsets_rejects_bad_mode() -> None:
    with pytest.raises(ValueError):
        count_onsets([0.1], threshold=0.1, mode="sideways")


def test_mean_closure_duration_averages_run_lengths() -> None:
    ear = [0.30, 0.05, 0.05, 0.30, 0.05]  # runs of length 2 and 1
    assert mean_closure_duration(ear, threshold=0.1) == pytest.approx(1.5)


def test_mean_closure_duration_no_closure_is_zero() -> None:
    assert mean_closure_duration([0.3, 0.3], threshold=0.1) == 0.0


def test_event_rate_per_minute() -> None:
    # 3 events over 300 frames at 30 fps is 10 seconds, so 18 per minute
    assert event_rate_per_minute(3, n_frames=300, fps=30.0) == pytest.approx(18.0)


def test_event_rate_zero_frames_is_zero() -> None:
    assert event_rate_per_minute(5, n_frames=0, fps=30.0) == 0.0


def test_summarize_window_returns_expected_features() -> None:
    ear = [0.30, 0.05, 0.05, 0.30, 0.30, 0.30]
    mar = [0.1, 0.1, 0.6, 0.6, 0.1, 0.1]
    pitch = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    feats = summarize_window(
        ear=ear, mar=mar, pitch=pitch, fps=30.0, ear_threshold=0.1, mar_threshold=0.5,
        nod_threshold=20.0,
    )
    assert feats["perclos"] == pytest.approx(2 / 6)
    assert feats["blink_count"] == 1
    assert feats["yawn_count"] == 1
    assert feats["nod_count"] == 0
    assert "mean_ear" in feats and "mean_mar" in feats
