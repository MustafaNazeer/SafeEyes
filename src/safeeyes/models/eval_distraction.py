"""Interval level evaluation for the distraction classifier.

Frames carry the model predictions, but the labeled unit is the interval. This
module aggregates per frame probability vectors into one vector per interval,
turns those into interval predictions, and reports honest metrics against a
fixed test split: overall and balanced accuracy, per class recall and support,
and a full confusion matrix. Balanced accuracy averages recall only over the
classes actually present in the labels, and the absent classes are named
explicitly so a sparse test split never inflates the reported number.

The functions here are pure and model independent. The injectable collection
seam takes any frame level predictor, so metric assembly is unit testable
without a trained checkpoint. Wiring a real backbone into this seam is left to
the evaluation console script.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from safeeyes.models.distraction_data import DISTRACTION_LABELS, FrameRecord

IntSeq = Sequence[int] | np.ndarray
N_CLASSES = len(DISTRACTION_LABELS)


def confusion_matrix(y_true: IntSeq, y_pred: IntSeq, n_classes: int) -> np.ndarray:
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true, pred, strict=True):
        matrix[t, p] += 1
    return matrix


def aggregate_interval_probs(frame_probs: np.ndarray) -> np.ndarray:
    array = np.asarray(frame_probs, dtype=float)
    if array.shape[0] == 0:
        raise ValueError("cannot aggregate an empty frame probability array")
    mean: np.ndarray = array.mean(axis=0)
    return mean


def evaluate_intervals(
    per_interval_probs: Sequence[np.ndarray], labels: Sequence[int]
) -> dict[str, object]:
    y_true = [int(label) for label in labels]
    y_pred = [int(np.argmax(np.asarray(probs))) for probs in per_interval_probs]
    cm = confusion_matrix(y_true, y_pred, n_classes=N_CLASSES)
    support = cm.sum(axis=1)
    correct = int(np.trace(cm))
    total = int(cm.sum())
    overall = correct / total if total else 0.0

    recalls = {
        DISTRACTION_LABELS[c]: (float(cm[c, c] / support[c]) if support[c] else 0.0)
        for c in range(N_CLASSES)
    }
    present = [c for c in range(N_CLASSES) if support[c] > 0]
    balanced = (
        float(np.mean([recalls[DISTRACTION_LABELS[c]] for c in present])) if present else 0.0
    )
    absent_classes = [DISTRACTION_LABELS[c] for c in range(N_CLASSES) if support[c] == 0]

    return {
        "overall_accuracy": overall,
        "balanced_accuracy": balanced,
        "per_class_recall": recalls,
        "support": {DISTRACTION_LABELS[c]: int(support[c]) for c in range(N_CLASSES)},
        "absent_classes": absent_classes,
        "confusion_matrix": {
            "labels": list(DISTRACTION_LABELS),
            "rows_true_cols_pred": cm.tolist(),
        },
        "n_classes": N_CLASSES,
    }


def collect_interval_probs(
    predict_frame_probs: Callable[[Sequence[Path]], np.ndarray],
    records: Sequence[FrameRecord],
) -> tuple[list[np.ndarray], list[int]]:
    order: list[str] = []
    groups: dict[str, list[FrameRecord]] = {}
    for record in records:
        if record.sample_id not in groups:
            groups[record.sample_id] = []
            order.append(record.sample_id)
        groups[record.sample_id].append(record)

    per_interval_probs: list[np.ndarray] = []
    per_interval_labels: list[int] = []
    for sample_id in order:
        group = groups[sample_id]
        label_index = group[0].label_index
        if any(record.label_index != label_index for record in group):
            raise ValueError(f"interval {sample_id!r} has mixed frame labels")
        frame_paths = [record.path for record in group]
        frame_probs = predict_frame_probs(frame_paths)
        per_interval_probs.append(aggregate_interval_probs(frame_probs))
        per_interval_labels.append(label_index)
    return per_interval_probs, per_interval_labels


def main(argv: Sequence[str] | None = None) -> int:
    import cv2
    import torch

    from safeeyes.models.distraction_data import frame_records
    from safeeyes.models.train_distraction import (
        build_model_from_checkpoint,
        build_transform,
    )

    parser = argparse.ArgumentParser(description="Evaluate the distraction classifier.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-manifest", nargs="+", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--metrics-out", default=None, help="optional metrics JSON path")
    args = parser.parse_args(argv)

    model, _ = build_model_from_checkpoint(args.checkpoint)
    transform = build_transform(train=False, size=args.size, normalize=True)

    def predict_frame_probs(frame_paths: Sequence[Path]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(frame_paths), args.batch_size):
            batch_paths = frame_paths[start : start + args.batch_size]
            tensors = []
            for path in batch_paths:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"could not read image: {path}")
                tensors.append(transform(image))
            batch = torch.stack(tensors)
            with torch.no_grad():
                probs = torch.softmax(model(batch), dim=1)
            chunks.append(probs.numpy())
        return np.concatenate(chunks, axis=0)

    records = frame_records(args.test_manifest, args.frames_root)
    per_interval_probs, labels = collect_interval_probs(predict_frame_probs, records)
    metrics = evaluate_intervals(per_interval_probs, labels)

    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    absent = metrics["absent_classes"]
    assert isinstance(absent, list)
    print(
        f"overall {metrics['overall_accuracy']:.4f} | "
        f"balanced {metrics['balanced_accuracy']:.4f} | "
        f"{len(absent)} classes absent from this split"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
