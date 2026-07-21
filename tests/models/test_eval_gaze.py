import numpy as np
import pytest

from safeeyes.models.eval_gaze import (
    binary_rates,
    interval_level_predictions,
)


def test_majority_vote_collapses_an_interval():
    ids = np.array([0, 0, 0, 1, 1])
    true = np.array(["front", "front", "front", "left", "left"])
    pred = np.array(["front", "front", "left", "left", "left"])
    got_true, got_pred = interval_level_predictions(ids, true, pred)
    assert got_true.tolist() == ["front", "left"]
    assert got_pred.tolist() == ["front", "left"]


def test_a_wrong_majority_is_reported_wrong():
    ids = np.array([0, 0, 0])
    true = np.array(["front", "front", "front"])
    pred = np.array(["left", "left", "front"])
    got_true, got_pred = interval_level_predictions(ids, true, pred)
    assert got_true.tolist() == ["front"]
    assert got_pred.tolist() == ["left"]


def test_intervals_are_returned_in_id_order():
    ids = np.array([2, 2, 0, 0, 1])
    true = np.array(["a", "a", "b", "b", "c"])
    pred = np.array(["a", "a", "b", "b", "c"])
    got_true, _ = interval_level_predictions(ids, true, pred)
    assert got_true.tolist() == ["b", "c", "a"]


def test_a_single_frame_interval_survives():
    ids = np.array([0])
    true = np.array(["front"])
    pred = np.array(["left"])
    got_true, got_pred = interval_level_predictions(ids, true, pred)
    assert (got_true.tolist(), got_pred.tolist()) == (["front"], ["left"])


def test_empty_input_yields_empty_output():
    got_true, got_pred = interval_level_predictions(
        np.array([], dtype=int), np.array([]), np.array([])
    )
    assert len(got_true) == len(got_pred) == 0


def test_a_true_label_that_varies_inside_an_interval_is_rejected():
    """An interval is one annotated glance, so its truth cannot change midway."""
    ids = np.array([0, 0])
    true = np.array(["front", "left"])
    pred = np.array(["front", "front"])
    with pytest.raises(ValueError, match="single true label"):
        interval_level_predictions(ids, true, pred)


def test_binary_rates_count_front_as_on_road():
    true = np.array(["front", "front", "left", "right"])
    pred = np.array(["front", "left", "left", "front"])
    rates = binary_rates(true, pred)
    # on road truth: 2 (indices 0, 1); one of them predicted off road -> FAR 1/2
    assert rates["n_on_road"] == 2
    assert rates["false_alarms"] == 1
    assert rates["false_alarm_rate"] == pytest.approx(0.5)
    # off road truth: 2 (indices 2, 3); one predicted off road -> detection 1/2
    assert rates["n_off_road"] == 2
    assert rates["detected"] == 1
    assert rates["detection_rate"] == pytest.approx(0.5)


def test_binary_rates_report_denominators_not_just_percentages():
    true = np.array(["front", "left"])
    pred = np.array(["front", "left"])
    rates = binary_rates(true, pred)
    assert set(rates) >= {
        "n_on_road",
        "n_off_road",
        "false_alarms",
        "detected",
        "false_alarm_rate",
        "detection_rate",
    }


def test_binary_rates_are_none_rather_than_zero_when_a_class_is_absent():
    """A rate with no denominator is undefined, not zero."""
    true = np.array(["left", "right"])
    pred = np.array(["left", "right"])
    rates = binary_rates(true, pred)
    assert rates["n_on_road"] == 0
    assert rates["false_alarm_rate"] is None
