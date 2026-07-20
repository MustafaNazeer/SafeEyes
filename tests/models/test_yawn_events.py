import numpy as np

from safeeyes.models.yawn_events import events_for_video, training_events

MIXED = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1])


def test_only_the_longest_event_is_positive_in_a_yawning_video():
    events = events_for_video("v.avi", "Female1", "Yawning", MIXED, 0.5)
    assert [(e.start, e.end, e.label) for e in events] == [(3, 6, 1)]


def test_all_events_in_a_talking_video_are_negative():
    events = events_for_video("v.avi", "Female1", "Talking", MIXED, 0.5)
    assert [e.label for e in events] == [0, 0]


def test_talking_and_yawning_video_is_treated_as_yawning():
    events = events_for_video("v.avi", "Female1", "Talking&Yawning", MIXED, 0.5)
    assert [e.label for e in events] == [1]


def test_duration_tie_breaks_toward_higher_peak_mar():
    mar = np.array([0.9, 0.1, 0.95, 0.1])
    events = events_for_video("v.avi", "Female1", "Yawning", mar, 0.5)
    assert events[0].start == 2


def test_a_yawning_video_with_no_event_contributes_nothing():
    events = events_for_video("v.avi", "Female1", "Yawning", np.array([0.1, 0.1]), 0.5)
    assert events == []


def test_remaining_tie_breaks_toward_earlier_start():
    mar = np.array([0.9, 0.1, 0.9])
    events = events_for_video("v.avi", "Female1", "Yawning", mar, 0.5)
    assert events[0].start == 0


def test_training_events_preserves_subject_ids():
    videos = [("a.avi", "Female1", "Yawning", MIXED), ("b.avi", "Male2", "Normal", MIXED)]
    events = training_events(videos, 0.5)
    assert {e.subject_id for e in events} == {"Female1", "Male2"}
