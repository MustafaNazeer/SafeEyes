"""Cross-dataset evaluation of the v1 temporal model on DMD drowsiness.

Loads the deployed temporal checkpoint, and for each extracted DMD face recording
aligns the per frame features with frame-level drowsy labels (sustained eye
closure), windows them the way the model consumes features, collapses the model's
three-class prediction to drowsy versus not, and reports binary detection metrics
aggregated over all recordings. Every recording is unseen by the model, so this is
a genuine cross-dataset generalization test.

Honesty, restated wherever the number appears: DMD drowsiness was acted, not real
fatigue, and the drowsy label is a sustained eye closure proxy derived from the
annotations, not an independent ground truth.

    python -m safeeyes.temporal.eval_cross_dmd \\
        --features-dir features/dmd-drowsiness \\
        --ann-dir data/dmd/drowsiness-ann \\
        --checkpoint models/edge/temporal.pt \\
        --out docs/ml/temporal-cross-dmd-metrics.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from safeeyes.data.dmd_drowsiness import (
    DEFAULT_FPS,
    DEFAULT_MIN_CLOSED_SECONDS,
    frame_drowsy_labels,
)
from safeeyes.temporal.cross_eval import evaluate_binary_drowsy, windowed_binary_labels


def subject_windows(
    features: np.ndarray,
    frame_indices: np.ndarray,
    n_raw_frames: int,
    annotation: dict[str, Any] | str | Path,
    *,
    window_size: int,
    stride: int,
    positive_fraction: float,
    fps: float = DEFAULT_FPS,
    min_closed_seconds: float = DEFAULT_MIN_CLOSED_SECONDS,
) -> tuple[list[np.ndarray], np.ndarray]:
    row_drowsy = frame_drowsy_labels(
        annotation, n_raw_frames, fps=fps, min_closed_seconds=min_closed_seconds
    )
    row_labels = row_drowsy[np.asarray(frame_indices, dtype=np.int64)]
    return windowed_binary_labels(features, row_labels, window_size, stride, positive_fraction)


def run(
    features_dir: str | Path,
    ann_dir: str | Path,
    checkpoint: str | Path,
    *,
    window_size: int = 150,
    stride: int = 75,
    positive_fraction: float = 0.1,
    fps: float = DEFAULT_FPS,
    min_closed_seconds: float = DEFAULT_MIN_CLOSED_SECONDS,
) -> dict[str, object]:
    import torch

    from safeeyes.alert.pipeline import make_gru_classifier
    from safeeyes.temporal.model import TemporalGRU

    features_dir = Path(features_dir)
    ann_dir = Path(ann_dir)

    model = TemporalGRU(n_features=5)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
    model.eval()
    classify = make_gru_classifier(model)

    all_windows: list[np.ndarray] = []
    all_true: list[bool] = []
    recordings = 0
    for npz_path in sorted(features_dir.glob("*.npz")):
        ann_path = ann_dir / f"{npz_path.stem}.json"
        if not ann_path.exists():
            continue
        data = np.load(npz_path)
        features = data["features"]
        frame_indices = data["frame_indices"]
        n_raw = int(data["n_raw_frames"])
        if features.shape[0] < window_size:
            continue
        windows, labels = subject_windows(
            features,
            frame_indices,
            n_raw,
            ann_path,
            window_size=window_size,
            stride=stride,
            positive_fraction=positive_fraction,
            fps=fps,
            min_closed_seconds=min_closed_seconds,
        )
        all_windows.extend(windows)
        all_true.extend(bool(v) for v in labels)
        recordings += 1

    metrics = evaluate_binary_drowsy(all_windows, all_true, classify)
    metrics["recordings"] = recordings
    metrics["config"] = {
        "window_size": window_size,
        "stride": stride,
        "positive_fraction": positive_fraction,
        "fps": fps,
        "min_closed_seconds": min_closed_seconds,
    }
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-dataset drowsy evaluation of the temporal model on DMD."
    )
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window", type=int, default=150)
    parser.add_argument("--stride", type=int, default=75)
    parser.add_argument("--positive-fraction", type=float, default=0.1)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--min-closed-seconds", type=float, default=DEFAULT_MIN_CLOSED_SECONDS)
    parser.add_argument("--out", default=None, help="write the metrics JSON here")
    args = parser.parse_args(argv)

    metrics = run(
        args.features_dir,
        args.ann_dir,
        args.checkpoint,
        window_size=args.window,
        stride=args.stride,
        positive_fraction=args.positive_fraction,
        fps=args.fps,
        min_closed_seconds=args.min_closed_seconds,
    )
    print(json.dumps(metrics, indent=2, default=float))
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
