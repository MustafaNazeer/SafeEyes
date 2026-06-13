"""Train and evaluate the temporal fatigue classifier on a subject independent split.

The reproducible harness: it windows each video's per frame feature sequence,
trains the model, and reports per class accuracy, macro AUROC, and the false
alarm rate on the held out subjects. Per frame features are produced by the
perception stage in a prior step and saved per video; this harness consumes
those feature arrays, so the reported numbers trace back to a fixed split and a
fixed feature extraction.

    python -m safeeyes.temporal.train_temporal \
        --train-manifest splits/uta-rldd/train.csv \
        --val-manifest splits/uta-rldd/test.csv \
        --feature-root features/uta-rldd --metrics-out docs/ml/temporal-metrics.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from safeeyes.data.manifest import read_manifest
from safeeyes.temporal.evaluation import evaluate_predictions
from safeeyes.temporal.model import GBTBaseline, TemporalGRU
from safeeyes.temporal.window import assemble_windowed_dataset

LabeledSequence = tuple[np.ndarray, int]
UTA_CLASS_TO_INDEX = {"alert": 0, "low_vigilance": 1, "drowsy": 2}


def train_and_evaluate(
    train_items: Sequence[LabeledSequence],
    val_items: Sequence[LabeledSequence],
    n_classes: int,
    window_size: int,
    stride: int,
    epochs: int,
    lr: float,
    seed: int = 0,
    alarm_class: int | None = None,
) -> dict[str, object]:
    torch.manual_seed(seed)
    x_train, y_train = assemble_windowed_dataset(train_items, window_size, stride)
    x_val, y_val = assemble_windowed_dataset(val_items, window_size, stride)

    model = TemporalGRU(n_features=x_train.shape[2], num_classes=n_classes)
    inputs = torch.tensor(x_train, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        criterion(model(inputs), targets).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_val, dtype=torch.float32))
        scores = torch.softmax(logits, dim=1).numpy()
        predictions = logits.argmax(dim=1).numpy()

    alarm = alarm_class if alarm_class is not None else n_classes - 1
    return evaluate_predictions(y_val, predictions, scores, n_classes, alarm)


def train_and_evaluate_gbt(
    train_items: Sequence[LabeledSequence],
    val_items: Sequence[LabeledSequence],
    n_classes: int,
    window_size: int,
    stride: int,
    seed: int = 0,
    alarm_class: int | None = None,
) -> dict[str, object]:
    """Gradient boosted trees baseline over flattened window features."""
    x_train, y_train = assemble_windowed_dataset(train_items, window_size, stride)
    x_val, y_val = assemble_windowed_dataset(val_items, window_size, stride)
    flat_train = x_train.reshape(x_train.shape[0], -1)
    flat_val = x_val.reshape(x_val.shape[0], -1)

    clf = GBTBaseline(random_state=seed).fit(flat_train, y_train)
    predictions = clf.predict(flat_val)
    scores = clf.predict_proba(flat_val)

    alarm = alarm_class if alarm_class is not None else n_classes - 1
    return evaluate_predictions(y_val, predictions, scores, n_classes, alarm)


def _load_items(manifest_path: str, feature_root: str) -> list[LabeledSequence]:
    root = Path(feature_root)
    items: list[LabeledSequence] = []
    for sample in read_manifest(manifest_path):
        feature_path = (root / sample.sample_id).with_suffix(".npy")
        sequence = np.load(feature_path)
        items.append((sequence, UTA_CLASS_TO_INDEX[sample.label]))
    return items


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the temporal fatigue classifier.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--feature-root", required=True, help="per video per frame feature arrays")
    parser.add_argument("--model", choices=["gru", "gbt"], default="gru")
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--window-size", type=int, default=150)
    parser.add_argument("--stride", type=int, default=75)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train_items = _load_items(args.train_manifest, args.feature_root)
    val_items = _load_items(args.val_manifest, args.feature_root)

    if args.model == "gbt":
        report = train_and_evaluate_gbt(
            train_items, val_items, n_classes=3, window_size=args.window_size, stride=args.stride,
            seed=args.seed,
        )
    else:
        report = train_and_evaluate(
            train_items, val_items, n_classes=3, window_size=args.window_size, stride=args.stride,
            epochs=args.epochs, lr=args.lr, seed=args.seed,
        )

    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
