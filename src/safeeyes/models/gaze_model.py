"""Gradient boosted gaze zone classifier and its leave one subject out run.

The classifier is deliberately plain: fixed hyperparameters, no threshold
tuning, no per fold selection. The binary eyes off road signal is the nine zone
argmax collapsed through is_eyes_on_road. With nothing tuned there is no inner
selection loop, so pooling the held out folds is a clean estimate rather than
an optimistic one.

The device runs onnxruntime alone and never gets scikit-learn, so the trained
model reaches the Pi through the skl2onnx conversion pinned in the edge tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from safeeyes.data.dmd_gaze import frame_interval_keys
from safeeyes.data.gaze_splits import subject_folds
from safeeyes.models.eval_gaze import (
    binary_rates,
    interval_level_predictions,
    zone_accuracy,
)


def train_gaze_model(x: np.ndarray, y: np.ndarray, seed: int = 0) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(x, y)
    return model


def load_corpus(
    feature_root: str | Path, gaze_root: str | Path | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """Load every recording, keyed by subject, with globally unique interval ids.

    When ``gaze_root`` is given, intervals are the annotated glances read back
    from the annotation files. That is the correct independent unit. The ids
    stored at extraction time instead break on any gap in detected frames,
    which splits one glance wherever face detection dropped a frame, and those
    dropouts cluster on the zones that turn the head away from the camera.

    Ids are offset per recording either way, since they restart at zero in each
    file and intervals from different people must never merge.
    """
    paths = sorted(Path(feature_root).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no gaze feature files under {feature_root}")
    annotations = _annotation_index(gaze_root) if gaze_root is not None else None

    by_subject: dict[str, list[dict[str, np.ndarray]]] = {}
    offset = 0
    for path in paths:
        # No allow_pickle: the archives store unicode labels, so loading a
        # feature file is never a code execution path.
        with np.load(path) as data:
            subject = str(data["subject"])
            frames = np.asarray(data["frame_indices"], dtype=int)
            ids = (
                _annotation_ids(frames, annotations[path.stem])
                if annotations is not None
                else np.asarray(data["interval_ids"], dtype=int)
            )
            record = {
                "features": np.asarray(data["features"], dtype=float),
                "labels": np.asarray(data["labels"]),
                "interval_ids": ids + offset,
            }
        offset += int(ids.max()) + 1 if ids.size else 0
        by_subject.setdefault(subject, []).append(record)

    merged: dict[str, dict[str, np.ndarray]] = {}
    for subject, records in by_subject.items():
        merged[subject] = {key: np.concatenate([r[key] for r in records]) for key in records[0]}
    return merged


def _annotation_index(gaze_root: str | Path) -> dict[str, dict]:
    """Annotation JSON for each recording, keyed by the feature file stem."""
    index: dict[str, dict] = {}
    for annotation_path in Path(gaze_root).rglob("*_rgb_ann_gaze.json"):
        stem = annotation_path.name.replace("_rgb_ann_gaze.json", "_rgb_face")
        with open(annotation_path) as handle:
            index[stem] = json.load(handle)
    if not index:
        raise FileNotFoundError(f"no gaze annotations under {gaze_root}")
    return index


def _annotation_ids(frames: np.ndarray, annotation: dict) -> np.ndarray:
    keys = frame_interval_keys(annotation, 10**9)
    dense: dict[str, int] = {}
    ids: list[int] = []
    for frame in frames.tolist():
        key = keys[frame]
        if key not in dense:
            dense[key] = len(dense)
        ids.append(dense[key])
    return np.array(ids, dtype=int)


def _stack(corpus: dict[str, dict[str, np.ndarray]], subjects: Sequence[str]) -> dict:
    return {
        key: np.concatenate([corpus[s][key] for s in subjects])
        for key in ("features", "labels", "interval_ids")
    }


def run_leave_one_subject_out(
    corpus: dict[str, dict[str, np.ndarray]], seed: int = 0
) -> dict[str, object]:
    folds = subject_folds(sorted(corpus))
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    pooled_ids: list[np.ndarray] = []
    per_subject: dict[str, object] = {}

    for train_subjects, held_out in folds:
        train = _stack(corpus, train_subjects)
        test = corpus[held_out]
        model = train_gaze_model(train["features"], train["labels"], seed=seed)
        predictions = model.predict(test["features"])

        pooled_true.append(test["labels"])
        pooled_pred.append(predictions)
        pooled_ids.append(test["interval_ids"])

        i_true, i_pred = interval_level_predictions(
            test["interval_ids"], test["labels"], predictions
        )
        per_subject[held_out] = {
            "frame": {
                "zone": zone_accuracy(test["labels"], predictions),
                "binary": binary_rates(test["labels"], predictions),
            },
            "interval": {
                "zone": zone_accuracy(i_true, i_pred),
                "binary": binary_rates(i_true, i_pred),
            },
        }
        print(f"  held out {held_out}: {len(test['labels'])} frames", flush=True)

    true = np.concatenate(pooled_true)
    pred = np.concatenate(pooled_pred)
    ids = np.concatenate(pooled_ids)
    i_true, i_pred = interval_level_predictions(ids, true, pred)

    return {
        "n_subjects": len(folds),
        "frame": {"zone": zone_accuracy(true, pred), "binary": binary_rates(true, pred)},
        "interval": {
            "zone": zone_accuracy(i_true, i_pred),
            "binary": binary_rates(i_true, i_pred),
        },
        "per_subject": per_subject,
        "_predictions": {"true": true, "pred": pred, "interval_ids": ids},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leave one subject out gaze evaluation.")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument(
        "--gaze-root",
        help="extracted DMD gaze tree; groups metrics by annotated glance",
    )
    parser.add_argument("--out", required=True, help="metrics JSON destination")
    parser.add_argument("--predictions-out", help="npz destination for held out predictions")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    corpus = load_corpus(args.feature_root, args.gaze_root)
    print(f"loaded {len(corpus)} subjects", flush=True)
    results = run_leave_one_subject_out(corpus, seed=args.seed)

    # Held out predictions are saved beside the metrics so the pooled result can
    # be regrouped or re-scored later without repeating the fourteen fold run.
    predictions = results.pop("_predictions")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    if args.predictions_out:
        predictions_path = Path(args.predictions_out)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(predictions_path, **predictions)
        print(f"wrote {predictions_path}")

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
