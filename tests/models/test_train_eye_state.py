from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from safeeyes.data.manifest import write_manifest
from safeeyes.data.splits import Sample
from safeeyes.models.train_eye_state import EyeStateDataset, evaluate


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
