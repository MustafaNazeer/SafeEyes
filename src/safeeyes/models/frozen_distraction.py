"""Frozen backbone feature caching and head only training for the distraction classifier.

Full fine tuning of the four pretrained backbones over every DMD frame is a multi
day job on a laptop CPU. This module takes the cheaper path: forward each backbone
over every frame once in eval mode, cache the feature vector the final linear layer
consumes, then train only that final linear on the cached features in seconds.

The design invariant is exact. The cached feature is the input to the final linear
of the model that the Task 5 registry builds, obtained by replacing that linear
(``fc`` for shufflenet_v2_x0_5, else ``classifier[-1]``) with an identity and
forwarding in eval mode. Training a fresh linear on those features and loading its
weights back into the final linear position reproduces the full model forward, so
the assembled checkpoint is byte compatible with what ``build_model_from_checkpoint``,
``safeeyes-eval-distraction``, and the edge export already expect.

Because features are cached in eval mode, the frozen dropout is off and the cached
values are deterministic, which also means the head trains on non augmented
features. That is an inherent tradeoff of caching, stated plainly rather than
hidden. The reported interval level metrics still come from ``safeeyes-eval-distraction``
run on the assembled checkpoint. The cache based frame accuracy this module prints
is only a fast sanity readout, never the reported number.

    python -m safeeyes.models.frozen_distraction \\
        --backbone mobilenet_v3_small \\
        --train-manifest splits/dmd/train.csv \\
        --val-manifest splits/dmd/val.csv \\
        --frames-root data/dmd/frames \\
        --cache-dir data/dmd/cache \\
        --out models/distraction.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from safeeyes.models.distraction_data import (
    DISTRACTION_LABELS,
    FrameDataset,
    frame_records,
)
from safeeyes.models.train_distraction import (
    BACKBONES,
    build_transform,
    default_size,
)

_HEAD_BATCH_SIZE = 256


def _final_linear(model: nn.Module, backbone: str) -> nn.Linear:
    module = cast(Any, model)
    head = module.fc if backbone == "shufflenet_v2_x0_5" else module.classifier[-1]
    if not isinstance(head, nn.Linear):
        raise TypeError(f"expected the final head of {backbone} to be nn.Linear")
    return head


def _set_final_linear(model: nn.Module, backbone: str, replacement: nn.Module) -> None:
    module = cast(Any, model)
    if backbone == "shufflenet_v2_x0_5":
        module.fc = replacement
    else:
        module.classifier[-1] = replacement


def feature_extractor(backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
    model = BACKBONES[backbone](len(DISTRACTION_LABELS), pretrained)
    dim = int(_final_linear(model, backbone).in_features)
    _set_final_linear(model, backbone, nn.Identity())
    model.eval()
    return model, dim


def cache_features(
    backbone: str,
    manifest_paths: Sequence[str | Path],
    frames_root: str | Path,
    out_path: str | Path,
    *,
    size: int,
    batch_size: int = 32,
    pretrained: bool = True,
) -> None:
    model, dim = feature_extractor(backbone, pretrained)
    records = frame_records(manifest_paths, frames_root)
    dataset = FrameDataset(records, build_transform(train=False, size=size, normalize=True))
    loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    batches: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, _ in loader:
            batches.append(model(inputs).cpu().numpy().astype(np.float32))

    if batches:
        features = np.concatenate(batches, axis=0)
    else:
        features = np.zeros((0, dim), dtype=np.float32)
    labels = np.array([record.label_index for record in records], dtype=np.int64)
    sample_ids = np.array([record.sample_id for record in records], dtype=np.str_)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, features=features, labels=labels, sample_ids=sample_ids)


def load_cache(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(str(path)) as data:
        return data["features"], data["labels"], data["sample_ids"]


def _clone_state(linear: nn.Linear) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in linear.state_dict().items()}


def train_head_on_cache(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    epochs: int,
    lr: float,
    seed: int,
    val: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    dim = int(x.shape[1])
    n = int(x.shape[0])

    linear = nn.Linear(dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr)

    val_x: torch.Tensor | None = None
    val_y: torch.Tensor | None = None
    if val is not None:
        val_x = torch.from_numpy(np.asarray(val[0], dtype=np.float32))
        val_y = torch.from_numpy(np.asarray(val[1], dtype=np.int64))

    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0
    losses: list[float] = []

    for _ in range(epochs):
        linear.train()
        order = torch.randperm(n)
        running = 0.0
        seen = 0
        for start in range(0, n, _HEAD_BATCH_SIZE):
            index = order[start : start + _HEAD_BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(linear(x[index]), y[index])
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * int(index.shape[0])
            seen += int(index.shape[0])
        losses.append(running / seen if seen else 0.0)

        if val_x is not None and val_y is not None:
            linear.eval()
            with torch.no_grad():
                predictions = linear(val_x).argmax(dim=1)
                accuracy = float((predictions == val_y).float().mean().item())
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = _clone_state(linear)

    if best_state is None:
        best_state = _clone_state(linear)
    return best_state, losses


def assemble_checkpoint(
    backbone: str,
    linear_state: dict[str, torch.Tensor],
    out_path: str | Path,
    *,
    pretrained: bool = True,
) -> None:
    num_classes = len(DISTRACTION_LABELS)
    model = BACKBONES[backbone](num_classes, pretrained)
    _final_linear(model, backbone).load_state_dict(linear_state)

    checkpoint = {
        "state_dict": model.state_dict(),
        "backbone": backbone,
        "labels": list(DISTRACTION_LABELS),
        "num_classes": num_classes,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)


def _cache_path(cache_dir: str | Path, backbone: str, split: str) -> Path:
    return Path(cache_dir) / backbone / f"{split}.npz"


def _ensure_cache(
    backbone: str,
    manifests: Sequence[str | Path],
    frames_root: str | Path,
    cache_path: Path,
    *,
    size: int,
    batch_size: int,
    pretrained: bool,
) -> None:
    if cache_path.exists():
        return
    cache_features(
        backbone,
        manifests,
        frames_root,
        cache_path,
        size=size,
        batch_size=batch_size,
        pretrained=pretrained,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cache frozen backbone features and train the distraction head."
    )
    parser.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    parser.add_argument("--train-manifest", nargs="+", required=True)
    parser.add_argument("--val-manifest", nargs="+", default=None)
    parser.add_argument("--test-manifest", nargs="+", default=None)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", required=True, help="checkpoint output path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)

    size = args.size if args.size is not None else default_size(args.backbone)
    num_classes = len(DISTRACTION_LABELS)

    def ensure(manifests: Sequence[str], split: str) -> Path:
        path = _cache_path(args.cache_dir, args.backbone, split)
        _ensure_cache(
            args.backbone,
            manifests,
            args.frames_root,
            path,
            size=size,
            batch_size=args.batch_size,
            pretrained=args.pretrained,
        )
        return path

    train_features, train_labels, _ = load_cache(ensure(args.train_manifest, "train"))

    val: tuple[np.ndarray, np.ndarray] | None = None
    if args.val_manifest:
        val_features, val_labels, _ = load_cache(ensure(args.val_manifest, "val"))
        val = (val_features, val_labels)

    linear_state, losses = train_head_on_cache(
        train_features,
        train_labels,
        num_classes=num_classes,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        val=val,
    )
    assemble_checkpoint(args.backbone, linear_state, args.out, pretrained=args.pretrained)

    summary = (
        f"backbone {args.backbone} | epochs {args.epochs} | "
        f"final train loss {losses[-1]:.4f} | checkpoint at {args.out}"
    )
    if args.test_manifest:
        test_features, test_labels, _ = load_cache(ensure(args.test_manifest, "test"))
        linear = nn.Linear(int(train_features.shape[1]), num_classes)
        linear.load_state_dict(linear_state)
        linear.eval()
        with torch.no_grad():
            predictions = linear(torch.from_numpy(test_features.astype(np.float32))).argmax(dim=1)
        truth = torch.from_numpy(test_labels.astype(np.int64))
        accuracy = float((predictions == truth).float().mean().item())
        summary += (
            f" | frozen head test frame accuracy {accuracy:.4f} "
            f"(cache sanity readout, not the reported metric)"
        )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
