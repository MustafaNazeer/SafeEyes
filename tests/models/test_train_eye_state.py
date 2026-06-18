from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from safeeyes.data.manifest import write_manifest
from safeeyes.data.splits import Sample
from safeeyes.models.train_eye_state import (
    EyeStateDataset,
    compute_class_weights,
    evaluate,
    train,
)


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


def test_dataset_length_matches_manifest(tmp_path: Path) -> None:
    manifest = _make_dataset(tmp_path / "imgs")
    ds = EyeStateDataset(manifest, tmp_path / "imgs", size=24)
    assert len(ds) == 4


def test_dataset_item_shape_and_label_mapping(tmp_path: Path) -> None:
    manifest = _make_dataset(tmp_path / "imgs")
    ds = EyeStateDataset(manifest, tmp_path / "imgs", size=24)
    tensor, label = ds[0]  # first sample is "open"
    assert tensor.shape == (1, 24, 24)
    assert label == 1  # open maps to 1, closed to 0
    _, closed_label = ds[1]
    assert closed_label == 0


def test_evaluate_computes_accuracy(tmp_path: Path) -> None:
    manifest = _make_dataset(tmp_path / "imgs")
    ds = EyeStateDataset(manifest, tmp_path / "imgs", size=24)
    loader = DataLoader(ds, batch_size=2)

    class AlwaysOpen(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(x.shape[0], 2)
            logits[:, 1] = 1.0  # always predict class 1 (open)
            return logits

    # Two of four samples are open, so always predicting open scores 0.5.
    assert evaluate(AlwaysOpen(), loader) == 0.5


def test_compute_class_weights_upweights_the_minority_class() -> None:
    # three closed (index 0), one open (index 1)
    weights = compute_class_weights([0, 0, 0, 1], n_classes=2)
    assert weights[1] > weights[0]
    assert weights.tolist() == pytest.approx([4 / 6, 4 / 2])


def test_compute_class_weights_uniform_when_balanced() -> None:
    weights = compute_class_weights([0, 1, 0, 1], n_classes=2)
    assert weights.tolist() == pytest.approx([1.0, 1.0])


def test_train_accepts_class_weights_and_runs(tmp_path: Path) -> None:
    manifest = _make_dataset(tmp_path / "imgs")
    ds = EyeStateDataset(manifest, tmp_path / "imgs", size=24)
    loader = DataLoader(ds, batch_size=2)
    from safeeyes.models.eye_state import EyeStateCNN

    model = EyeStateCNN()
    weights = compute_class_weights([1, 0, 1, 0], n_classes=2)
    train(model, loader, epochs=1, lr=0.01, class_weights=weights)
    # a forward pass still produces two-class logits after weighted training
    tensor, _ = ds[0]
    assert model(tensor.unsqueeze(0)).shape == (1, 2)
