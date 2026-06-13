import numpy as np

from safeeyes.alert.pipeline import DrowsinessPipeline, make_gru_classifier
from safeeyes.alert.state_machine import AlertTier
from safeeyes.temporal.model import TemporalGRU


def test_no_alert_before_window_is_full() -> None:
    pipe = DrowsinessPipeline(
        classifier=lambda w: 2, window_capacity=4, n_features=3, escalate_steps=3
    )
    tier = pipe.process([0.0, 0.0, 0.0])
    assert tier == AlertTier.NONE


def test_classifier_runs_only_once_the_window_is_full() -> None:
    calls = {"n": 0}

    def classifier(window: np.ndarray) -> int:
        calls["n"] += 1
        return 0

    pipe = DrowsinessPipeline(
        classifier=classifier, window_capacity=3, n_features=2, escalate_steps=2
    )
    for _ in range(5):
        pipe.process([0.0, 0.0])
    # window fills on frame 3, classifier runs on frames 3, 4, 5
    assert calls["n"] == 3


def test_sustained_drowsy_escalates_to_audible() -> None:
    pipe = DrowsinessPipeline(
        classifier=lambda w: 2, window_capacity=2, n_features=1, escalate_steps=3, alarm_after=100
    )
    tier = AlertTier.NONE
    for _ in range(8):
        tier = pipe.process([0.5])
    assert tier == AlertTier.AUDIBLE


def test_window_is_passed_to_classifier_with_expected_shape() -> None:
    seen = {}

    def classifier(window: np.ndarray) -> int:
        seen["shape"] = window.shape
        return 0

    pipe = DrowsinessPipeline(
        classifier=classifier, window_capacity=4, n_features=3, escalate_steps=2
    )
    for _ in range(4):
        pipe.process([1.0, 2.0, 3.0])
    assert seen["shape"] == (4, 3)


def test_make_gru_classifier_returns_a_valid_class_index() -> None:
    model = TemporalGRU(n_features=3, num_classes=3)
    classify = make_gru_classifier(model)
    level = classify(np.zeros((5, 3)))
    assert isinstance(level, int)
    assert level in (0, 1, 2)
