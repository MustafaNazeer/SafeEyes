from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from safeeyes.data.yawdd_crops import extract_clip_crops, extract_manifest_crops
from safeeyes.models.yawn_events import YawnEvent, proposal_events, training_events
from safeeyes.models.yawn_model import (
    MAX_SUBSTITUTION_ROWS,
    _event_crops,
    _nearest_available_row,
    build_event_features,
    event_feature_vector,
    sample_event_rows,
    score_events,
    train_yawn_head,
)
from safeeyes.temporal.yawn_validation import CROP_MARGIN_STEPS


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


def test_event_feature_vector_pools_mean_and_max_across_crops():
    # Each row stands in for one crop's backbone feature vector. The values are
    # chosen so that pooling across crops (axis 0, the correct axis) and
    # pooling across the feature dimension (axis 1, the mutated axis) land on
    # different numbers despite the square shape, so a transposed pooling axis
    # cannot slip through with a coincidentally matching result.
    crops = np.array(
        [
            [1, 2, 3, 4, 5],
            [10, 1, 1, 1, 1],
            [1, 20, 1, 1, 1],
            [1, 1, 30, 1, 1],
            [1, 1, 1, 1, 40],
        ],
        dtype=np.float32,
    )
    backbone = nn.Identity()

    def stub_transform(crop: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(crop)

    vector = event_feature_vector(crops, backbone, stub_transform)

    expected_mean = crops.mean(axis=0)
    expected_max = crops.max(axis=0)
    assert vector.shape == (10,)
    np.testing.assert_allclose(vector[:5], expected_mean)
    np.testing.assert_allclose(vector[5:], expected_max)


def test_substitution_bound_stays_coupled_to_the_crop_extraction_margin():
    # MAX_SUBSTITUTION_ROWS is meant to track yawdd_crops.py's margin_steps
    # default exactly, not merely start out equal to it. Reading the real
    # default off extract_clip_crops and extract_manifest_crops (rather than
    # repeating the number 5 here) means this fails if either module ever
    # falls back to a bare literal instead of the shared CROP_MARGIN_STEPS.
    assert MAX_SUBSTITUTION_ROWS == CROP_MARGIN_STEPS
    for fn in (extract_clip_crops, extract_manifest_crops):
        assert inspect.signature(fn).parameters["margin_steps"].default == MAX_SUBSTITUTION_ROWS


def test_event_features_default_to_the_leak_free_proposal_builder():
    # The label blind proposal_events is the automatic default so that omitting
    # the keyword fails safe: a caller that forgets it gets the rule a detector
    # can actually run at test time, never the label aware training_events that
    # reads a video's ground truth to decide which openings count. Read off the
    # signature rather than exercised through a run, because the property under
    # guard is which builder an omitted keyword resolves to.
    default = inspect.signature(build_event_features).parameters["events_builder"].default
    assert default is proposal_events
    assert default is not training_events


def test_nearest_available_row_returns_the_exact_row_when_present():
    crop_rows = np.array([3, 7, 12])
    assert _nearest_available_row(7, crop_rows) == 7


def test_nearest_available_row_substitutes_within_the_bound():
    crop_rows = np.array([5, 20])
    assert _nearest_available_row(7, crop_rows) == 5
    assert abs(5 - 7) <= MAX_SUBSTITUTION_ROWS


def test_nearest_available_row_accepts_a_substitution_exactly_at_the_bound():
    # The bound is a maximum, not a strict upper limit: a substitution exactly
    # MAX_SUBSTITUTION_ROWS away is still legitimate and must be accepted, not
    # rejected. This is the boundary the raise below sits one row past.
    crop_rows = np.array([0, 20])
    row = MAX_SUBSTITUTION_ROWS
    assert _nearest_available_row(row, crop_rows) == 0


def test_nearest_available_row_raises_past_the_bound():
    crop_rows = np.array([0, 20])
    row = MAX_SUBSTITUTION_ROWS + 1
    with pytest.raises(ValueError, match="exceeding the maximum acceptable substitution"):
        _nearest_available_row(row, crop_rows)


def test_nearest_available_row_raises_with_no_crops_at_all():
    with pytest.raises(ValueError, match="no crops are available"):
        _nearest_available_row(0, np.array([], dtype=int))


def test_event_crops_uses_the_exact_row_when_available():
    crop_rows = np.array([0, 1, 2, 3, 4])
    crops = np.arange(5).reshape(5, 1, 1, 1).astype(np.uint8)
    event = YawnEvent(sample_id="s", subject_id="p", start=0, end=4, peak_mar=0.9, label=1)
    result = _event_crops(event, crop_rows, crops, n_frames=5)
    assert result[:, 0, 0, 0].tolist() == [0, 1, 2, 3, 4]


def test_event_crops_substitutes_the_nearest_row_within_the_bound():
    crop_rows = np.array([0, 4])
    crops = np.array([[[[0]]], [[[4]]]], dtype=np.uint8)
    event = YawnEvent(sample_id="s", subject_id="p", start=0, end=4, peak_mar=0.9, label=1)
    # sample_event_rows(0, 4, n=5) is [0, 1, 2, 3, 4]; rows 1 to 3 fall back to
    # the nearer of the two available rows (a tie at row 2 breaks toward the
    # first, lower, candidate), each within MAX_SUBSTITUTION_ROWS.
    result = _event_crops(event, crop_rows, crops, n_frames=5)
    assert result[:, 0, 0, 0].tolist() == [0, 0, 0, 4, 4]


def test_event_crops_raises_when_the_only_substitution_is_too_far():
    far = MAX_SUBSTITUTION_ROWS + 1
    crop_rows = np.array([0])
    crops = np.array([[[[0]]]], dtype=np.uint8)
    event = YawnEvent(sample_id="s", subject_id="p", start=far, end=far, peak_mar=0.9, label=1)
    with pytest.raises(ValueError, match="exceeding the maximum acceptable substitution"):
        _event_crops(event, crop_rows, crops, n_frames=1)
