import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from safeeyes.models.distraction_data import DISTRACTION_LABELS, LABEL_TO_INDEX, FrameRecord
from safeeyes.models.eval_distraction import (
    aggregate_interval_probs,
    collect_interval_probs,
    confusion_matrix,
    evaluate_intervals,
)

N_CLASSES = 13


def _one_hot(index: int) -> np.ndarray:
    vec = np.zeros(N_CLASSES, dtype=float)
    vec[index] = 1.0
    return vec


def test_aggregate_interval_probs_is_frame_mean() -> None:
    frame_probs = np.array(
        [
            [0.2, 0.8] + [0.0] * (N_CLASSES - 2),
            [0.6, 0.4] + [0.0] * (N_CLASSES - 2),
        ]
    )
    aggregated = aggregate_interval_probs(frame_probs)
    assert aggregated.shape == (N_CLASSES,)
    assert aggregated[0] == pytest.approx(0.4)
    assert aggregated[1] == pytest.approx(0.6)


def test_aggregate_interval_probs_empty_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_interval_probs(np.zeros((0, N_CLASSES)))


def test_confusion_matrix_counts_true_by_predicted() -> None:
    y_true = [8, 8, 1, 1, 1]
    y_pred = [8, 1, 1, 1, 8]
    cm = confusion_matrix(y_true, y_pred, n_classes=N_CLASSES)
    assert cm[8, 8] == 1
    assert cm[8, 1] == 1
    assert cm[1, 1] == 2
    assert cm[1, 8] == 1
    assert cm.sum() == 5


def test_evaluate_intervals_overall_and_balanced_with_absent_classes() -> None:
    drinking = LABEL_TO_INDEX["drinking"]
    safe_drive = LABEL_TO_INDEX["safe_drive"]
    per_interval_probs = [
        _one_hot(safe_drive),
        _one_hot(safe_drive),
        _one_hot(drinking),
        _one_hot(safe_drive),
    ]
    labels = [safe_drive, safe_drive, drinking, drinking]
    metrics = evaluate_intervals(per_interval_probs, labels)

    assert metrics["overall_accuracy"] == pytest.approx(0.75)
    assert metrics["n_classes"] == N_CLASSES

    recall = metrics["per_class_recall"]
    assert isinstance(recall, dict)
    assert recall["safe_drive"] == pytest.approx(1.0)
    assert recall["drinking"] == pytest.approx(0.5)

    assert metrics["balanced_accuracy"] == pytest.approx((1.0 + 0.5) / 2)

    support = metrics["support"]
    assert isinstance(support, dict)
    assert support["safe_drive"] == 2
    assert support["drinking"] == 2
    assert support["change_gear"] == 0

    absent = metrics["absent_classes"]
    assert isinstance(absent, list)
    assert "change_gear" in absent
    assert "standstill_or_waiting" in absent
    assert "safe_drive" not in absent
    assert "drinking" not in absent


def test_evaluate_intervals_confusion_matrix_shape_and_labels() -> None:
    drinking = LABEL_TO_INDEX["drinking"]
    safe_drive = LABEL_TO_INDEX["safe_drive"]
    per_interval_probs = [
        _one_hot(safe_drive),
        _one_hot(drinking),
        _one_hot(safe_drive),
    ]
    labels = [safe_drive, drinking, drinking]
    metrics = evaluate_intervals(per_interval_probs, labels)

    cm = metrics["confusion_matrix"]
    assert isinstance(cm, dict)
    assert cm["labels"] == list(DISTRACTION_LABELS)
    rows = cm["rows_true_cols_pred"]
    assert len(rows) == N_CLASSES
    assert all(len(row) == N_CLASSES for row in rows)
    assert rows[safe_drive][safe_drive] == 1
    assert rows[drinking][drinking] == 1
    assert rows[drinking][safe_drive] == 1


def test_evaluate_intervals_degenerate_single_prediction_no_nans() -> None:
    safe_drive = LABEL_TO_INDEX["safe_drive"]
    drinking = LABEL_TO_INDEX["drinking"]
    radio = LABEL_TO_INDEX["radio"]
    per_interval_probs = [_one_hot(safe_drive) for _ in range(3)]
    labels = [safe_drive, drinking, radio]
    metrics = evaluate_intervals(per_interval_probs, labels)

    assert metrics["overall_accuracy"] == pytest.approx(1 / 3)
    recall = metrics["per_class_recall"]
    assert isinstance(recall, dict)
    assert recall["safe_drive"] == pytest.approx(1.0)
    assert recall["drinking"] == pytest.approx(0.0)
    assert recall["radio"] == pytest.approx(0.0)
    assert metrics["balanced_accuracy"] == pytest.approx((1.0 + 0.0 + 0.0) / 3)

    for value in recall.values():
        assert not np.isnan(value)
    assert not np.isnan(metrics["balanced_accuracy"])
    assert not np.isnan(metrics["overall_accuracy"])


def test_evaluate_intervals_metrics_json_round_trip() -> None:
    safe_drive = LABEL_TO_INDEX["safe_drive"]
    drinking = LABEL_TO_INDEX["drinking"]
    metrics = evaluate_intervals(
        [_one_hot(safe_drive), _one_hot(drinking)],
        [safe_drive, drinking],
    )
    restored = json.loads(json.dumps(metrics))
    assert restored == metrics


def _record(sample_id: str, label_index: int, frame: int) -> FrameRecord:
    return FrameRecord(
        path=Path(f"{sample_id}/frame_{frame:05d}.jpg"),
        label_index=label_index,
        sample_id=sample_id,
    )


def test_collect_interval_probs_groups_by_sample_and_aggregates() -> None:
    safe_drive = LABEL_TO_INDEX["safe_drive"]
    drinking = LABEL_TO_INDEX["drinking"]
    records = [
        _record("a", safe_drive, 0),
        _record("a", safe_drive, 1),
        _record("b", drinking, 0),
        _record("a", safe_drive, 2),
    ]

    seen_groups: list[list[Path]] = []

    def stub_predict(frame_paths: Sequence[Path]) -> np.ndarray:
        seen_groups.append(list(frame_paths))
        return np.tile(_one_hot(safe_drive), (len(frame_paths), 1))

    per_interval_probs, per_interval_labels = collect_interval_probs(stub_predict, records)

    assert per_interval_labels == [safe_drive, drinking]
    assert len(per_interval_probs) == 2
    assert [len(group) for group in seen_groups] == [3, 1]
    assert per_interval_probs[0].shape == (N_CLASSES,)
    assert per_interval_probs[0][safe_drive] == pytest.approx(1.0)


def test_collect_interval_probs_mixed_labels_raises() -> None:
    safe_drive = LABEL_TO_INDEX["safe_drive"]
    drinking = LABEL_TO_INDEX["drinking"]
    records = [
        _record("a", safe_drive, 0),
        _record("a", drinking, 1),
    ]

    def stub_predict(frame_paths: Sequence[Path]) -> np.ndarray:
        return np.tile(_one_hot(safe_drive), (len(frame_paths), 1))

    with pytest.raises(ValueError):
        collect_interval_probs(stub_predict, records)
