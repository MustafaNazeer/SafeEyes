"""tests/alert/test_replay.py"""
import numpy as np

from safeeyes.alert.replay import TierEvent, classify_sequence, replay_levels
from safeeyes.alert.state_machine import AlertStateMachine, AlertTier


def test_classify_sequence_is_zero_until_window_fills_then_tracks_classifier():
    calls = []

    def classifier(window):
        calls.append(window.copy())
        return 2

    features = np.ones((7, 5))
    levels = classify_sequence(features, classifier, window_size=5)
    assert levels.tolist() == [0, 0, 0, 0, 2, 2, 2]
    assert len(calls) == 3
    assert calls[0].shape == (5, 5)


def test_classify_sequence_shorter_than_window_stays_level_zero():
    def classifier(window):
        raise AssertionError("classifier must not run on a partial window")

    levels = classify_sequence(np.ones((3, 5)), classifier, window_size=5)
    assert levels.tolist() == [0, 0, 0]


def test_classify_sequence_empty_input():
    levels = classify_sequence(np.empty((0, 5)), lambda w: 2, window_size=5)
    assert levels.tolist() == []


def test_replay_levels_emits_only_transitions_with_frame_indices():
    machine = AlertStateMachine(escalate_steps=2, de_escalate_steps=3, alarm_after=100)
    levels = [0, 0, 2, 2, 2, 0, 0, 0, 0]
    events = replay_levels(levels, machine)
    assert events == [
        TierEvent(frame=3, tier=AlertTier.AUDIBLE),
        TierEvent(frame=7, tier=AlertTier.NONE),
    ]


def test_replay_levels_resets_machine_between_clips():
    machine = AlertStateMachine(escalate_steps=1, de_escalate_steps=1, alarm_after=100)
    replay_levels([2, 2, 2], machine)
    events = replay_levels([0, 0], machine)
    assert events == []
