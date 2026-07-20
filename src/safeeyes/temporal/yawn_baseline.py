"""Duration sweep for the geometric MAR yawn baseline.

The threshold rule alone (MAR crossing MAR_YAWN_THRESHOLD at least once) reaches
high recall but low precision on the YawDD Mirror set, because ordinary talking
crosses the threshold constantly. A real yawn opening lasts far longer than a
talking opening, so this module sweeps a minimum event duration on top of the
fixed threshold and reports the video level precision and recall at each one.

This sweep is run on train subjects only. The selection rule is fixed: among
the durations whose recall is at least the floor, take the highest precision,
ties going to the smaller duration. If no duration reaches the floor, the
selected duration is the one with the highest recall, and the sweep records
that the floor was not met.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from safeeyes.data.manifest import read_manifest
from safeeyes.data.yawdd import is_yawning
from safeeyes.temporal.yawn_validation import MAR_YAWN_THRESHOLD, video_predicts_yawning

Video = tuple[str, str, np.ndarray]


def load_mirror_mar(manifest_path: str | Path, feature_root: str | Path) -> list[Video]:
    feature_root = Path(feature_root)
    videos: list[Video] = []
    for sample in read_manifest(manifest_path):
        feature_path = (feature_root / sample.sample_id).with_suffix(".npy")
        if not feature_path.exists():
            raise FileNotFoundError(f"missing feature array: {feature_path}")
        features = np.load(feature_path)
        mar = features[:, 1]
        videos.append((sample.sample_id, sample.label, mar))
    return videos


def precision_recall(
    videos: Sequence[Video], threshold: float, min_duration: int
) -> tuple[float, float, int, int, int]:
    tp = fp = fn = 0
    for _sample_id, label, mar in videos:
        truth = is_yawning(label.split("&"))
        predicted = video_predicts_yawning(mar, threshold, min_duration)
        if truth and predicted:
            tp += 1
        elif truth:
            fn += 1
        elif predicted:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, tp, fp, fn


def sweep_min_duration(
    videos: Sequence[Video],
    threshold: float,
    durations: Iterable[int],
    min_recall: float = 0.90,
) -> dict[str, object]:
    measurements: list[tuple[int, float, float, int, int, int]] = []
    for d in durations:
        precision, recall, tp, fp, fn = precision_recall(videos, threshold, d)
        measurements.append((d, precision, recall, tp, fp, fn))

    rows: list[dict[str, object]] = [
        {
            "min_duration": d,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        for d, precision, recall, tp, fp, fn in measurements
    ]

    eligible = [m for m in measurements if m[2] >= min_recall]
    floor_met = bool(eligible)
    if floor_met:
        selected = max(eligible, key=lambda m: (m[1], -m[0]))[0]
    else:
        selected = max(measurements, key=lambda m: (m[2], -m[0]))[0]

    return {"selected": selected, "floor_met": floor_met, "rows": rows}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep the minimum event duration for the geometric MAR yawn baseline."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--threshold", type=float, default=MAR_YAWN_THRESHOLD)
    parser.add_argument("--max-duration", type=int, default=20)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    videos = load_mirror_mar(args.manifest, args.feature_root)
    result = sweep_min_duration(
        videos,
        args.threshold,
        durations=range(1, args.max_duration + 1),
        min_recall=args.min_recall,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(args.manifest),
        "feature_root": str(args.feature_root),
        "threshold": args.threshold,
        "max_duration": args.max_duration,
        "min_recall": args.min_recall,
        **result,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
