from __future__ import annotations

import numpy as np

from safeeyes.models.yawn_model import sample_event_rows, score_events, train_yawn_head


def test_event_rows_are_evenly_spread():
    assert sample_event_rows(10, 18, n=5) == [10, 12, 14, 16, 18]


def test_short_events_repeat_rows():
    # Python's round is banker's rounding, so round(0.5) is 0 and round(0.75) is 1.
    # The expectation below is what the formula actually produces, not what an
    # even spread would suggest. Repeating a row is harmless here.
    assert sample_event_rows(4, 5, n=5) == [4, 4, 4, 5, 5]


def test_single_row_event_repeats_one_row():
    assert sample_event_rows(7, 7, n=5) == [7, 7, 7, 7, 7]


def test_head_learns_separable_features():
    rng = np.random.default_rng(0)
    positive = rng.normal(2.0, 0.2, size=(40, 8)).astype(np.float32)
    negative = rng.normal(-2.0, 0.2, size=(40, 8)).astype(np.float32)
    features = np.vstack([positive, negative])
    labels = np.array([1] * 40 + [0] * 40)
    head = train_yawn_head(features, labels, epochs=50, seed=0)
    scores = score_events(head, features)
    assert scores[:40].mean() > scores[40:].mean() + 0.3


def test_scores_are_probabilities():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(10, 8)).astype(np.float32)
    head = train_yawn_head(features, np.array([1, 0] * 5), epochs=5, seed=0)
    scores = score_events(head, features)
    assert scores.shape == (10,)
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0


def test_training_is_deterministic_under_a_seed():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, 8)).astype(np.float32)
    labels = np.array([1, 0] * 10)
    first = score_events(train_yawn_head(features, labels, epochs=10, seed=0), features)
    second = score_events(train_yawn_head(features, labels, epochs=10, seed=0), features)
    assert np.allclose(first, second)
