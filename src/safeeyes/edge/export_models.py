"""Turn trained checkpoints into edge artifacts.

Loads a PyTorch checkpoint, exports it to ONNX, and (by default) writes an int8
quantized copy alongside it. This is the laptop side of the edge workflow: run it
once after training, copy the resulting ``.int8.onnx`` files to the Pi, and point
the runtime at them.

    python -m safeeyes.edge.export_models \
        --temporal-checkpoint models/temporal.pt --n-features 8 --out-dir models/edge
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from safeeyes.edge.export import (
    export_distraction_onnx,
    export_eye_state_onnx,
    export_temporal_onnx,
)
from safeeyes.edge.quantize import quantize_dynamic_onnx
from safeeyes.models.eye_state import EyeStateCNN
from safeeyes.temporal.model import TemporalGRU


@dataclass(frozen=True)
class Artifacts:
    onnx: Path
    quantized: Path | None


def export_and_quantize_temporal(
    checkpoint: str | Path,
    out_dir: str | Path,
    n_features: int,
    n_classes: int = 3,
    quantize: bool = True,
) -> Artifacts:
    model = TemporalGRU(n_features=n_features, num_classes=n_classes)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
    model.eval()
    onnx_path = export_temporal_onnx(model, Path(out_dir) / "temporal.onnx", n_features=n_features)
    return _maybe_quantize(onnx_path, quantize)


def export_and_quantize_eye(
    checkpoint: str | Path,
    out_dir: str | Path,
    crop_size: int = 24,
    quantize: bool = True,
) -> Artifacts:
    model = EyeStateCNN()
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
    model.eval()
    onnx_path = export_eye_state_onnx(model, Path(out_dir) / "eye_state.onnx", crop_size=crop_size)
    return _maybe_quantize(onnx_path, quantize)


def export_and_quantize_distraction(
    checkpoint: str | Path,
    out_dir: str | Path,
    *,
    size: int = 224,
    quantize: bool = True,
) -> Artifacts:
    from safeeyes.models.train_distraction import build_model_from_checkpoint

    record = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    backbone = str(record["backbone"])
    model, _labels = build_model_from_checkpoint(checkpoint)
    onnx_path = export_distraction_onnx(
        model, Path(out_dir) / f"distraction_{backbone}.onnx", size=size
    )
    return _maybe_quantize(onnx_path, quantize)


def _maybe_quantize(onnx_path: Path, quantize: bool) -> Artifacts:
    if not quantize:
        return Artifacts(onnx=onnx_path, quantized=None)
    quantized = quantize_dynamic_onnx(onnx_path, onnx_path.with_suffix(".int8.onnx"))
    return Artifacts(onnx=onnx_path, quantized=quantized)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export trained checkpoints to ONNX edge artifacts."
    )
    parser.add_argument("--temporal-checkpoint", help="trained TemporalGRU checkpoint")
    parser.add_argument("--eye-checkpoint", help="trained EyeStateCNN checkpoint")
    parser.add_argument(
        "--distraction-checkpoint",
        nargs="+",
        help="one or more trained distraction backbone checkpoints",
    )
    parser.add_argument("--out-dir", required=True, help="directory to write the artifacts into")
    parser.add_argument("--n-features", type=int, default=8, help="temporal feature count")
    parser.add_argument("--n-classes", type=int, default=3, help="temporal fatigue classes")
    parser.add_argument("--crop-size", type=int, default=24, help="eye crop side length")
    parser.add_argument("--distraction-size", type=int, default=224, help="distraction input side")
    parser.add_argument("--no-quantize", action="store_true", help="export ONNX only, skip int8")
    args = parser.parse_args(argv)

    if not args.temporal_checkpoint and not args.eye_checkpoint and not args.distraction_checkpoint:
        parser.error(
            "provide --temporal-checkpoint, --eye-checkpoint, --distraction-checkpoint, or a mix"
        )

    quantize = not args.no_quantize
    if args.temporal_checkpoint:
        export_and_quantize_temporal(
            args.temporal_checkpoint, args.out_dir, args.n_features, args.n_classes, quantize
        )
    if args.eye_checkpoint:
        export_and_quantize_eye(args.eye_checkpoint, args.out_dir, args.crop_size, quantize)
    if args.distraction_checkpoint:
        for checkpoint in args.distraction_checkpoint:
            export_and_quantize_distraction(
                checkpoint, args.out_dir, size=args.distraction_size, quantize=quantize
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
