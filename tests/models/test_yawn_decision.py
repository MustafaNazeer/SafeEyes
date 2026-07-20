import numpy as np

from safeeyes.models.yawn_decision import select_tau, video_scores
from safeeyes.models.yawn_events import YawnEvent


def _event(sample_id):
    return YawnEvent(
        sample_id=sample_id, subject_id="Female1", start=0, end=3, peak_mar=0.9, label=0
    )


def test_video_score_is_the_maximum_event_score():
    events = [_event("a.avi"), _event("a.avi"), _event("b.avi")]
    scores = np.array([0.2, 0.8, 0.4])
    assert video_scores(events, scores) == {"a.avi": 0.8, "b.avi": 0.4}


def test_a_video_with_no_event_scores_zero():
    assert video_scores([], np.array([])) == {}


def test_tau_selection_prefers_precision_above_the_recall_floor():
    scores = {"y1": 0.9, "y2": 0.8, "t1": 0.4, "n1": 0.1}
    truths = {"y1": True, "y2": True, "t1": False, "n1": False}
    result = select_tau(scores, truths, taus=[0.05, 0.5, 0.85], min_recall=0.9)
    assert result["selected"] == 0.5
    assert result["floor_met"] is True


def test_tau_selection_reports_a_missed_floor():
    scores = {"y1": 0.1, "y2": 0.1}
    truths = {"y1": True, "y2": True}
    result = select_tau(scores, truths, taus=[0.5, 0.9], min_recall=0.9)
    assert result["floor_met"] is False


def test_a_video_with_no_proposed_event_never_fires():
    # y1 has no entry in video_scores at all (no proposed event), so it must be
    # treated as scoring 0.0 and can never clear a positive tau. This is the
    # inherited recall ceiling the brief describes, not a missing key bug.
    scores = {"y2": 0.9}
    truths = {"y1": True, "y2": True}
    result = select_tau(scores, truths, taus=[0.1], min_recall=0.9)
    assert result["floor_met"] is False


def test_tau_selection_breaks_a_genuine_precision_tie_toward_the_lower_tau():
    # Both taus below yield identical (tp, fp, fn) = (2, 0, 0), so precision and
    # recall are exactly equal at 0.3 and 0.6. Only the tie break can decide
    # between them, so this fails if the max() key ever drops its tau term.
    scores = {"y1": 0.9, "y2": 0.9, "n1": 0.1}
    truths = {"y1": True, "y2": True, "n1": False}
    result = select_tau(scores, truths, taus=[0.6, 0.3], min_recall=0.9)
    assert result["selected"] == 0.3


def test_tau_selection_treats_recall_exactly_at_the_floor_as_met():
    # 9 of 10 positives score above tau, exactly the 0.9 floor. A mutation from
    # >= to > on the recall comparison flips this case and only this case.
    scores = {f"y{i}": 0.9 for i in range(9)}
    scores["y9"] = 0.1
    truths = {f"y{i}": True for i in range(10)}
    result = select_tau(scores, truths, taus=[0.5], min_recall=0.9)
    assert result["floor_met"] is True
    assert result["selected"] == 0.5
