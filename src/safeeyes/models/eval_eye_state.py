"""Evaluate a trained eye state checkpoint on a fixed subject independent split.

Separated from training so the reported numbers can be regenerated from a saved
checkpoint at any time, without retraining. The held out accuracy, the per class
recall, and the confusion matrix all come from this one pass over the fixed test
manifest, so every figure in the model card traces back to a script and a split
rather than to a remembered value.

    python -m safeeyes.models.eval_eye_state \
        --checkpoint models/eye_state.pt \
        --manifest splits/mrl-eye/test.csv \
        --image-root data/mrl-eye/mrlEyes_2018_01 \
        --metrics-out docs/ml/eye-state-metrics.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from safeeyes.models.eye_state import EyeStateCNN
from safeeyes.models.train_eye_state import EyeStateDataset

_INDEX_TO_LABEL = {0: "closed", 1: "open"}

IntSeq = Sequence[int] | np.ndarray


def confusion_matrix(y_true: IntSeq, y_pred: IntSeq, n_classes: int) -> np.ndarray:
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true, pred, strict=True):
        matrix[t, p] += 1
    return matrix


def eye_state_metrics(y_true: IntSeq, y_pred: IntSeq) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, n_classes=2)
    support = cm.sum(axis=1)
    correct = int(np.trace(cm))
    total = int(cm.sum())
    overall = correct / total if total else 0.0
    recalls = {
        _INDEX_TO_LABEL[c]: (float(cm[c, c] / support[c]) if support[c] else 0.0) for c in range(2)
    }
    balanced = float(np.mean(list(recalls.values())))
    return {
        "overall_accuracy": overall,
        "balanced_accuracy": balanced,
        "per_class_recall": recalls,
        "support": {_INDEX_TO_LABEL[c]: int(support[c]) for c in range(2)},
        "confusion_matrix": {
            "labels": ["closed", "open"],
            "rows_true_cols_pred": cm.tolist(),
        },
    }


def collect_predictions(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]]
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            predictions = model(inputs).argmax(dim=1)
            y_pred.extend(int(p) for p in predictions.tolist())
            y_true.extend(int(t) for t in labels.tolist())
    return y_true, y_pred


def evaluate_checkpoint(
    checkpoint: str | Path,
    manifest: str | Path,
    image_root: str | Path,
    size: int = 24,
    batch_size: int = 256,
) -> dict[str, object]:
    model = EyeStateCNN()
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
    dataset = EyeStateDataset(manifest, image_root, size=size)
    loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(dataset, batch_size=batch_size)
    y_true, y_pred = collect_predictions(model, loader)
    return eye_state_metrics(y_true, y_pred)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the eye state classifier.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="test split manifest")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--metrics-out", default=None, help="optional metrics JSON path")
    args = parser.parse_args(argv)

    metrics = evaluate_checkpoint(
        args.checkpoint, args.manifest, args.image_root, size=args.size, batch_size=args.batch_size
    )
    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    recall = metrics["per_class_recall"]
    assert isinstance(recall, dict)
    print(
        f"overall {metrics['overall_accuracy']:.4f} | "
        f"balanced {metrics['balanced_accuracy']:.4f} | "
        f"closed {recall['closed']:.4f} | open {recall['open']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
