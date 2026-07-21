"""Train the deployment gaze model and export it to ONNX in one step.

The device runs onnxruntime alone and never gets scikit-learn, so the trained
classifier reaches it as ONNX. Training and conversion happen together and only
the ONNX artifact is written: there is no intermediate pickle, so no part of
the pipeline ever loads one, and loading a model is never a code execution
path.

The deployed model trains on all fourteen subjects. The reported numbers come
from the leave one subject out run, where every subject was held out in turn,
so the figures describe this architecture and feature set rather than this
exact artifact. That is stated in the model card rather than papered over.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from safeeyes.models.gaze_model import load_corpus, train_gaze_model
from safeeyes.perception.gaze_features import GAZE_FEATURE_COLUMNS


def export_gaze_onnx(model, path: str | Path, n_features: int) -> Path:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    onnx_model = convert_sklearn(
        model, initial_types=[("input", FloatTensorType([None, n_features]))]
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onnx_model.SerializeToString())
    return out


def verify_parity(model, onnx_path: str | Path, probe: np.ndarray) -> int:
    """Return the number of rows where the ONNX artifact and the model disagree."""
    from safeeyes.edge.runtime import OnnxModel

    session = OnnxModel(onnx_path)
    onnx_labels = np.asarray(session.run(probe.astype(np.float32))).ravel()
    return int((onnx_labels != model.predict(probe)).sum())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and export the gaze zone model.")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--gaze-root")
    parser.add_argument("--out", required=True, help="ONNX destination")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    corpus = load_corpus(args.feature_root, args.gaze_root)
    features = np.concatenate([corpus[s]["features"] for s in sorted(corpus)])
    labels = np.concatenate([corpus[s]["labels"] for s in sorted(corpus)])
    print(f"training on {len(corpus)} subjects, {len(features)} rows", flush=True)

    model = train_gaze_model(features, labels, seed=args.seed)
    out = export_gaze_onnx(model, args.out, len(GAZE_FEATURE_COLUMNS))

    mismatches = verify_parity(model, out, features)
    print(f"wrote {out}")
    print(f"parity mismatches over {len(features)} rows: {mismatches}")
    if mismatches:
        raise SystemExit(f"ONNX export disagrees with the trained model on {mismatches} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
