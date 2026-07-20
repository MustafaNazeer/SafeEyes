from __future__ import annotations

import numpy as np
import pytest

from safeeyes.models.eval_yawn import (
    PRECISION_BAR,
    RECALL_DROP_ALLOWANCE,
    RECALL_FLOOR,
    deploy_decision,
    score_detector,
)
from safeeyes.models.yawn_events import proposal_events


def test_deploy_requires_both_conditions():
    assert deploy_decision(cnn=(0.75, 0.92), baseline=(0.60, 0.95)) == "deploy"


def test_deploy_blocked_below_the_precision_bar():
    assert deploy_decision(cnn=(0.65, 0.95), baseline=(0.60, 0.95)) == "blocked_precision"


def test_deploy_blocked_below_the_recall_floor():
    assert deploy_decision(cnn=(0.90, 0.85), baseline=(0.60, 0.95)) == "blocked_recall"


def test_deploy_blocked_when_it_does_not_beat_the_baseline():
    assert deploy_decision(cnn=(0.72, 0.92), baseline=(0.80, 0.93)) == "blocked_baseline"


def test_deploy_blocked_when_recall_falls_more_than_five_points_below_baseline():
    assert deploy_decision(cnn=(0.95, 0.90), baseline=(0.60, 0.99)) == "blocked_baseline"


def test_precision_bar_is_inclusive_at_exactly_the_bar():
    assert deploy_decision(cnn=(PRECISION_BAR, 0.95), baseline=(0.60, 0.95)) == "deploy"


def test_precision_just_below_the_bar_is_blocked():
    assert deploy_decision(cnn=(PRECISION_BAR - 0.01, 0.95), baseline=(0.10, 0.95)) == (
        "blocked_precision"
    )


def test_recall_floor_is_inclusive_at_exactly_the_floor():
    assert deploy_decision(cnn=(0.95, RECALL_FLOOR), baseline=(0.60, 0.92)) == "deploy"


def test_recall_just_below_the_floor_is_blocked():
    assert deploy_decision(cnn=(0.95, RECALL_FLOOR - 0.01), baseline=(0.60, 0.90)) == (
        "blocked_recall"
    )


def test_equal_precision_does_not_beat_the_baseline():
    assert deploy_decision(cnn=(0.80, 0.95), baseline=(0.80, 0.95)) == "blocked_baseline"


def test_precision_bar_is_checked_before_the_baseline_comparison():
    # Precision clears neither the absolute bar nor the baseline. The absolute
    # bar is the first gate, so the reported reason must be the precision bar
    # rather than the baseline comparison.
    assert deploy_decision(cnn=(0.50, 0.95), baseline=(0.60, 0.95)) == "blocked_precision"


def test_recall_floor_is_checked_before_the_baseline_comparison():
    assert deploy_decision(cnn=(0.95, 0.80), baseline=(0.99, 0.82)) == "blocked_recall"


def test_recall_exactly_the_allowance_below_the_baseline_still_deploys():
    baseline_recall = 0.99
    assert (
        deploy_decision(
            cnn=(0.95, baseline_recall - RECALL_DROP_ALLOWANCE),
            baseline=(0.60, baseline_recall),
        )
        == "deploy"
    )


def test_recall_a_hair_more_than_the_allowance_below_the_baseline_is_blocked():
    baseline_recall = 0.99
    assert (
        deploy_decision(
            cnn=(0.95, baseline_recall - RECALL_DROP_ALLOWANCE - 0.001),
            baseline=(0.60, baseline_recall),
        )
        == "blocked_baseline"
    )


def _video(sample_id, label, mar):
    return (sample_id, label, np.asarray(mar, dtype=float))


def test_score_detector_reports_precision_recall_and_talking_false_positives():
    videos = [
        _video("y1", "Yawning", [0.0, 1.0, 1.0, 0.0]),
        _video("y2", "Talking&Yawning", [0.0, 0.0, 0.0]),
        _video("t1", "Talking", [1.0, 1.0, 0.0]),
        _video("n1", "Normal", [0.0, 0.0]),
    ]
    result = score_detector(videos, threshold=0.5, min_duration=1)
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["fp"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    # One non yawning Talking video, and it fired.
    assert result["talking_false_positive_rate"] == pytest.approx(1.0)


def test_score_detector_honours_the_minimum_duration():
    videos = [
        _video("y1", "Yawning", [1.0, 1.0, 1.0, 1.0]),
        _video("t1", "Talking", [1.0, 0.0, 1.0, 0.0]),
    ]
    result = score_detector(videos, threshold=0.5, min_duration=3)
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["talking_false_positive_rate"] == pytest.approx(0.0)


def test_score_detector_counts_a_silent_video_as_a_false_negative():
    videos = [_video("y1", "Yawning", [0.0, 0.0, 0.0])]
    result = score_detector(videos, threshold=0.5, min_duration=1)
    assert result["tp"] == 0
    assert result["fn"] == 1
    assert result["recall"] == pytest.approx(0.0)


def test_proposal_events_ignores_the_video_label():
    mar = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    labelled = proposal_events([("v.avi", "Female1", "Yawning", mar)], threshold=0.5)
    unlabelled = proposal_events([("v.avi", "Female1", "Normal", mar)], threshold=0.5)
    # training_events would keep only the longest run of a Yawning video. An
    # inference time proposal has no label to consult, so both videos must
    # yield the same two events.
    assert [(e.start, e.end) for e in labelled] == [(0, 1), (3, 5)]
    assert [(e.start, e.end) for e in labelled] == [(e.start, e.end) for e in unlabelled]
