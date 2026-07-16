from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from safeeyes.data.intervals import IntervalSample, sanitize_sample_id, write_interval_manifest
from safeeyes.models.distraction_data import DISTRACTION_LABELS, frame_records
from safeeyes.models.frozen_distraction import (
    assemble_checkpoint,
    cache_features,
    feature_extractor,
    load_cache,
    train_head_on_cache,
)
from safeeyes.models.train_distraction import build_model_from_checkpoint, default_size

TORCHVISION_BACKBONES = [
    "mobilenet_v3_small",
    "mobilenet_v2",
    "efficientnet_b0",
    "shufflenet_v2_x0_5",
]


def _write_solid_frame(path: Path, color_bgr: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :] = color_bgr
    cv2.imwrite(str(path), image)


def _write_frames(frames_root: Path, sample_id: str, count: int) -> None:
    directory = frames_root / sanitize_sample_id(sample_id)
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        cv2.imwrite(
            str(directory / f"frame_{index:06d}.jpg"),
            np.full((32, 32, 3), 128, dtype=np.uint8),
        )


@pytest.mark.parametrize("backbone", TORCHVISION_BACKBONES)
def test_feature_extractor_shape_and_dim(backbone: str) -> None:
    model, dim = feature_extractor(backbone, False)
    size = default_size(backbone)
    with torch.no_grad():
        features = model(torch.randn(2, 3, size, size))
    assert features.ndim == 2
    assert features.shape == (2, dim)


def test_cache_round_trip(tmp_path: Path) -> None:
    backbone = "shufflenet_v2_x0_5"
    frames_root = tmp_path / "frames"
    samples = [
        IntervalSample(
            sample_id="a_rgb_body.mp4#0-10",
            subject_id="gA_1",
            label="safe_drive",
            start_frame=0,
            end_frame=10,
        ),
        IntervalSample(
            sample_id="b_rgb_body.mp4#5-9",
            subject_id="gA_2",
            label="drinking",
            start_frame=5,
            end_frame=9,
        ),
    ]
    _write_frames(frames_root, samples[0].sample_id, 3)
    _write_frames(frames_root, samples[1].sample_id, 2)
    manifest = tmp_path / "train.csv"
    write_interval_manifest(samples, manifest)

    records = frame_records([manifest], frames_root)
    out = tmp_path / "cache" / backbone / "train.npz"
    _, dim = feature_extractor(backbone, False)

    cache_features(
        backbone,
        [manifest],
        frames_root,
        out,
        size=default_size(backbone),
        batch_size=2,
        pretrained=False,
    )

    features, labels, sample_ids = load_cache(out)
    n = len(records)
    assert features.shape == (n, dim)
    assert features.dtype == np.float32
    assert labels.tolist() == [r.label_index for r in records]
    assert list(sample_ids) == [r.sample_id for r in records]


def test_train_head_reduces_loss_and_selects_best_val() -> None:
    rng = np.random.default_rng(0)
    per_class = 40
    dim = 8
    cluster0 = rng.normal(-2.0, 0.1, size=(per_class, dim)).astype(np.float32)
    cluster1 = rng.normal(2.0, 0.1, size=(per_class, dim)).astype(np.float32)
    features = np.concatenate([cluster0, cluster1], axis=0)
    labels = np.array([0] * per_class + [1] * per_class, dtype=np.int64)

    val0 = rng.normal(-2.0, 0.1, size=(10, dim)).astype(np.float32)
    val1 = rng.normal(2.0, 0.1, size=(10, dim)).astype(np.float32)
    val_features = np.concatenate([val0, val1], axis=0)
    val_labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    state, losses = train_head_on_cache(
        features,
        labels,
        num_classes=2,
        epochs=12,
        lr=1e-1,
        seed=0,
        val=(val_features, val_labels),
    )

    assert len(losses) == 12
    assert losses[-1] < losses[0]

    linear = nn.Linear(dim, 2)
    linear.load_state_dict(state)
    linear.eval()
    with torch.no_grad():
        predictions = linear(torch.from_numpy(val_features)).argmax(dim=1)
    accuracy = float((predictions == torch.from_numpy(val_labels)).float().mean().item())
    assert accuracy == 1.0


@pytest.mark.parametrize("backbone", ["mobilenet_v3_small", "shufflenet_v2_x0_5"])
def test_assembly_equivalence(tmp_path: Path, backbone: str) -> None:
    num_classes = len(DISTRACTION_LABELS)

    torch.manual_seed(1234)
    extractor, dim = feature_extractor(backbone, False)

    torch.manual_seed(999)
    linear = nn.Linear(dim, num_classes)
    linear_state = {name: value.detach().clone() for name, value in linear.state_dict().items()}

    torch.manual_seed(1234)
    out = tmp_path / "assembled.pt"
    assemble_checkpoint(backbone, linear_state, out, pretrained=False)

    model, labels = build_model_from_checkpoint(out)
    assert labels == list(DISTRACTION_LABELS)

    size = default_size(backbone)
    image = torch.randn(3, 3, size, size)
    with torch.no_grad():
        expected = linear(extractor(image))
        actual = model(image)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_assembled_checkpoint_reloads_in_task5_format(tmp_path: Path) -> None:
    backbone = "mobilenet_v3_small"
    num_classes = len(DISTRACTION_LABELS)
    _, dim = feature_extractor(backbone, False)
    linear = nn.Linear(dim, num_classes)
    linear_state = {name: value.detach().clone() for name, value in linear.state_dict().items()}

    out = tmp_path / "assembled.pt"
    assemble_checkpoint(backbone, linear_state, out, pretrained=False)

    model, labels = build_model_from_checkpoint(out)
    assert labels == list(DISTRACTION_LABELS)
    size = default_size(backbone)
    with torch.no_grad():
        logits = model(torch.randn(4, 3, size, size))
    assert logits.shape == (4, num_classes)
