"""Train and evaluate the eye state classifier on the MRL Eye split.

Run on the laptop once the MRL split manifests exist (see docs/source for how to
build them). The split is subject independent, so the reported held out accuracy
reflects performance on people the model never trained on. Training writes a
checkpoint and a small metrics file so the reported number traces back to a
specific run and split.

    python -m safeeyes.models.train_eye_state \
        --train-manifest splits/mrl-eye/train.csv \
        --val-manifest splits/mrl-eye/test.csv \
        --image-root data/mrl-eye --out models/eye_state.pt
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from safeeyes.data.manifest import read_manifest
from safeeyes.models.eye_state import EyeStateCNN, preprocess_eye

_LABEL_TO_INDEX = {"closed": 0, "open": 1}


class EyeStateDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, manifest_path: str | Path, image_root: str | Path, size: int = 24) -> None:
        self._samples = read_manifest(manifest_path)
        self._root = Path(image_root)
        self._size = size

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self._samples[index]
        image = cv2.imread(str(self._root / sample.sample_id), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"could not read image: {self._root / sample.sample_id}")
        tensor = preprocess_eye(image, size=self._size)
        return tensor, _LABEL_TO_INDEX[sample.label]


def evaluate(model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]]) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            predictions = model(inputs).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
    return correct / total if total else 0.0


def train(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    epochs: int,
    lr: float,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the eye state classifier.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", required=True, help="checkpoint output path")
    parser.add_argument("--metrics-out", default=None, help="optional metrics JSON path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)

    train_ds = EyeStateDataset(args.train_manifest, args.image_root, size=args.size)
    val_ds = EyeStateDataset(args.val_manifest, args.image_root, size=args.size)
    train_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        val_ds, batch_size=args.batch_size
    )

    model = EyeStateCNN()
    train(model, train_loader, epochs=args.epochs, lr=args.lr)
    accuracy = evaluate(model, val_loader)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)

    metrics = {"held_out_accuracy": accuracy, "val_samples": len(val_ds), "epochs": args.epochs}
    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"held out accuracy: {accuracy:.4f} on {len(val_ds)} samples; checkpoint at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
