from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from safeeyes.data.manifest import write_manifest
from safeeyes.data.splits import Sample
from safeeyes.models.eval_eye_state import (
    collect_predictions,
    confusion_matrix,
    eye_state_metrics,
)
from safeeyes.models.train_eye_state import EyeStateDataset


def test_confusion_matrix_counts_true_by_predicted() -> None:
    # index 0 is closed, index 1 is open. Rows are the true label, columns predicted.
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    cm = confusion_matrix(y_true, y_pred, n_classes=2)
    assert cm.tolist() == [[1, 1], [1, 2]]


def test_confusion_matrix_all_correct_is_diagonal() -> None:
    cm = confusion_matrix([0, 1, 1], [0, 1, 1], n_classes=2)
    assert cm.tolist() == [[1, 0], [0, 2]]


def test_eye_state_metrics_overall_and_per_class_recall() -> None:
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    report = eye_state_metrics(y_true, y_pred)
    assert report["overall_accuracy"] == pytest.approx(0.6)
    assert report["per_class_recall"]["closed"] == pytest.approx(0.5)
    assert report["per_class_recall"]["open"] == pytest.approx(2 / 3)


def test_eye_state_metrics_balanced_accuracy_is_mean_of_recalls() -> None:
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    report = eye_state_metrics(y_true, y_pred)
    assert report["balanced_accuracy"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_eye_state_metrics_records_confusion_and_support() -> None:
    report = eye_state_metrics([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    assert report["confusion_matrix"]["labels"] == ["closed", "open"]
    assert report["confusion_matrix"]["rows_true_cols_pred"] == [[1, 1], [1, 2]]
    assert report["support"] == {"closed": 2, "open": 3}


def test_eye_state_metrics_perfect_classifier() -> None:
    report = eye_state_metrics([0, 1, 1], [0, 1, 1])
    assert report["overall_accuracy"] == pytest.approx(1.0)
    assert report["balanced_accuracy"] == pytest.approx(1.0)


def _make_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, label in enumerate(["open", "closed", "open", "closed"]):
        name = f"img_{i}.png"
        value = 200 if label == "open" else 20
        cv2.imwrite(str(root / name), np.full((30, 40), value, dtype=np.uint8))
        samples.append(Sample(sample_id=name, subject_id=f"s{i}", label=label))
    manifest = root / "manifest.csv"
    write_manifest(samples, manifest)
    return manifest


def test_collect_predictions_returns_true_and_predicted_labels(tmp_path: Path) -> None:
    manifest = _make_dataset(tmp_path / "imgs")
    ds = EyeStateDataset(manifest, tmp_path / "imgs", size=24)
    loader = DataLoader(ds, batch_size=2)

    class AlwaysOpen(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(x.shape[0], 2)
            logits[:, 1] = 1.0
            return logits

    y_true, y_pred = collect_predictions(AlwaysOpen(), loader)
    # manifest order is open, closed, open, closed -> labels 1, 0, 1, 0
    assert y_true == [1, 0, 1, 0]
    assert y_pred == [1, 1, 1, 1]
