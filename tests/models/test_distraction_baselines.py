from pathlib import Path

import numpy as np
import pytest
import torch

from safeeyes.models.distraction_baselines import (
    DistractionScratchCNN,
    fit_majority_class,
    majority_frame_predictor,
)
from safeeyes.models.distraction_data import LABEL_TO_INDEX, FrameRecord
from safeeyes.models.eval_distraction import (
    N_CLASSES,
    collect_interval_probs,
    evaluate_intervals,
)


def _record(sample_id: str, label_index: int, frame: int) -> FrameRecord:
    return FrameRecord(
        path=Path(f"{sample_id}/frame_{frame:05d}.jpg"),
        label_index=label_index,
        sample_id=sample_id,
    )


def test_fit_majority_class_picks_most_common_label() -> None:
    safe_drive = LABEL_TO_INDEX["safe_drive"]
    drinking = LABEL_TO_INDEX["drinking"]
    records = [
        _record("a", safe_drive, 0),
        _record("b", drinking, 0),
        _record("c", safe_drive, 0),
        _record("d", safe_drive, 0),
    ]
    assert fit_majority_class(records) == safe_drive


def test_fit_majority_class_breaks_ties_by_lowest_index() -> None:
    lower = min(LABEL_TO_INDEX["drinking"], LABEL_TO_INDEX["safe_drive"])
    higher = max(LABEL_TO_INDEX["drinking"], LABEL_TO_INDEX["safe_drive"])
    records = [
        _record("a", higher, 0),
        _record("b", lower, 0),
        _record("c", higher, 0),
        _record("d", lower, 0),
    ]
    assert fit_majority_class(records) == lower


def test_fit_majority_class_empty_raises() -> None:
    with pytest.raises(ValueError):
        fit_majority_class([])


def test_majority_frame_predictor_returns_one_hot_block() -> None:
    majority = LABEL_TO_INDEX["safe_drive"]
    predict = majority_frame_predictor(majority)
    paths = [Path(f"frame_{i:05d}.jpg") for i in range(5)]
    probs = predict(paths)

    assert probs.shape == (5, N_CLASSES)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0.0)
    assert np.all(np.argmax(probs, axis=1) == majority)


def test_majority_frame_predictor_empty_paths_returns_empty_block() -> None:
    majority = LABEL_TO_INDEX["safe_drive"]
    predict = majority_frame_predictor(majority)
    probs = predict([])
    assert probs.shape == (0, N_CLASSES)


def test_majority_baseline_flows_through_shared_interval_path() -> None:
    safe_drive = LABEL_TO_INDEX["safe_drive"]
    drinking = LABEL_TO_INDEX["drinking"]
    records = [
        _record("a", safe_drive, 0),
        _record("a", safe_drive, 1),
        _record("b", drinking, 0),
        _record("c", safe_drive, 0),
    ]
    majority = fit_majority_class(records)
    assert majority == safe_drive

    predict = majority_frame_predictor(majority)
    per_interval_probs, labels = collect_interval_probs(predict, records)
    metrics = evaluate_intervals(per_interval_probs, labels)

    predictions = [int(np.argmax(probs)) for probs in per_interval_probs]
    assert predictions == [safe_drive, safe_drive, safe_drive]
    assert labels == [safe_drive, drinking, safe_drive]
    assert metrics["overall_accuracy"] == pytest.approx(2 / 3)


def test_scratch_cnn_forward_shape_contract() -> None:
    model = DistractionScratchCNN()
    logits = model(torch.zeros(2, 3, 96, 96))
    assert logits.shape == (2, N_CLASSES)


def test_scratch_cnn_is_input_size_agnostic() -> None:
    model = DistractionScratchCNN()
    logits = model(torch.zeros(2, 3, 128, 128))
    assert logits.shape == (2, N_CLASSES)


def test_scratch_cnn_respects_num_classes_argument() -> None:
    model = DistractionScratchCNN(num_classes=5)
    logits = model(torch.zeros(3, 3, 96, 96))
    assert logits.shape == (3, 5)
