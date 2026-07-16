from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from safeeyes.models.distraction_data import DISTRACTION_LABELS, FrameDataset, FrameRecord
from safeeyes.models.train_distraction import (
    BACKBONES,
    build_model_from_checkpoint,
    build_transform,
    head_parameters,
    train_distraction,
)

N_CLASSES = len(DISTRACTION_LABELS)


def _head_out_features(model: torch.nn.Module, backbone: str) -> int:
    if backbone == "shufflenet_v2_x0_5":
        return int(model.fc.out_features)
    return int(model.classifier[-1].out_features)


@pytest.mark.parametrize("backbone", sorted(BACKBONES))
def test_every_backbone_builds_with_thirteen_class_head(backbone: str) -> None:
    model = BACKBONES[backbone](N_CLASSES, False)
    assert _head_out_features(model, backbone) == N_CLASSES


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    backbone = "scratch"
    model = BACKBONES[backbone](N_CLASSES, False)
    checkpoint = {
        "state_dict": model.state_dict(),
        "backbone": backbone,
        "labels": list(DISTRACTION_LABELS),
        "num_classes": N_CLASSES,
    }
    out = tmp_path / "distraction.pt"
    torch.save(checkpoint, out)

    restored, labels = build_model_from_checkpoint(out)
    assert labels == list(DISTRACTION_LABELS)
    with torch.no_grad():
        logits = restored(torch.zeros(1, 3, 96, 96))
    assert logits.shape == (1, N_CLASSES)


def test_train_transform_has_no_flip() -> None:
    transform = build_transform(train=True, size=96, normalize=True)
    names = {type(step).__name__ for step in transform.augmentations}
    assert "RandomHorizontalFlip" not in names
    assert "RandomVerticalFlip" not in names
    assert names, "expected at least one augmentation on the train transform"


def test_transform_converts_bgr_to_rgb() -> None:
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255
    transform = build_transform(train=False, size=8, normalize=False)
    tensor = transform(bgr)
    assert tensor.shape == (3, 8, 8)
    assert float(tensor[0].mean()) == pytest.approx(0.0)
    assert float(tensor[2].mean()) == pytest.approx(1.0)


def _write_solid_frame(path: Path, color_bgr: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :] = color_bgr
    cv2.imwrite(str(path), image)


def _color_separable_records(tmp_path: Path, per_class: int = 6) -> list[FrameRecord]:
    palette = {0: (255, 0, 0), 1: (0, 0, 255)}
    records: list[FrameRecord] = []
    for label_index, color in palette.items():
        for i in range(per_class):
            sample_id = f"c{label_index}_{i}"
            frame = tmp_path / sample_id / "frame_00000.jpg"
            _write_solid_frame(frame, color)
            records.append(
                FrameRecord(path=frame, label_index=label_index, sample_id=sample_id)
            )
    return records


def test_training_reduces_loss_on_separable_toy(tmp_path: Path) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    records = _color_separable_records(tmp_path)
    transform = build_transform(train=True, size=96, normalize=True)
    dataset = FrameDataset(records, transform)
    loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        dataset, batch_size=4, shuffle=True
    )
    model = BACKBONES["scratch"](N_CLASSES, False)
    losses = train_distraction(
        model, loader, epochs=8, lr=1e-3, backbone="scratch"
    )
    assert losses[-1] < losses[0]


def test_freeze_backbone_keeps_backbone_fixed_and_trains_head(tmp_path: Path) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    backbone = "mobilenet_v3_small"
    records = _color_separable_records(tmp_path, per_class=4)
    transform = build_transform(train=False, size=224, normalize=True)
    dataset = FrameDataset(records, transform)
    loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(dataset, batch_size=4)
    model = BACKBONES[backbone](N_CLASSES, False)

    backbone_weight = model.features[0][0].weight
    head_weight = model.classifier[-1].weight
    backbone_before = backbone_weight.detach().clone()
    head_before = head_weight.detach().clone()

    train_distraction(
        model, loader, epochs=1, lr=1e-2, backbone=backbone, freeze_backbone=True
    )

    assert torch.equal(backbone_weight.detach(), backbone_before)
    assert not torch.equal(head_weight.detach(), head_before)


def test_head_parameters_selects_only_the_head() -> None:
    backbone = "mobilenet_v3_small"
    model = BACKBONES[backbone](N_CLASSES, False)
    head_ids = {id(p) for p in head_parameters(model, backbone)}
    expected = {id(p) for p in model.classifier[-1].parameters()}
    assert head_ids == expected
