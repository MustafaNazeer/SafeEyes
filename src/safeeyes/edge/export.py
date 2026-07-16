"""Export trained PyTorch models to ONNX for the edge runtime.

The two models take different export paths because of one architectural detail.
The eye state CNN ends in an adaptive average pool whose output size is not an
integer factor of its input, which the older TorchScript ONNX path rejects, so
it goes through the graph-capture exporter. The temporal GRU exports cleanly on
the TorchScript path with dynamic batch and dynamic sequence length. Each model
uses the exporter that reproduces it faithfully; both are verified against the
PyTorch forward pass in the tests.

The eye state CNN is exported with a dynamic batch dimension so a frame can
classify both eyes in one call. The temporal GRU is exported with dynamic batch
and dynamic sequence length so the same graph serves whatever window size the
runtime settles on.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch import nn

from safeeyes.models.eye_state import EyeStateCNN
from safeeyes.temporal.model import TemporalGRU

_OPSET = 17


def export_eye_state_onnx(
    model: EyeStateCNN,
    path: str | Path,
    crop_size: int = 24,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 1, crop_size, crop_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        torch.onnx.export(
            model.eval(),
            (dummy,),
            str(out),
            input_names=["eye"],
            output_names=["logits"],
            dynamic_axes={"eye": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=True,
            verbose=False,
        )
    return out


def export_temporal_onnx(
    model: TemporalGRU,
    path: str | Path,
    n_features: int,
    seq_len: int = 150,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, seq_len, n_features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        torch.onnx.export(
            model.eval(),
            (dummy,),
            str(out),
            input_names=["window"],
            output_names=["logits"],
            dynamic_axes={"window": {0: "batch", 1: "frames"}, "logits": {0: "batch"}},
            opset_version=_OPSET,
            dynamo=False,
        )
    return out


def export_distraction_onnx(
    model: nn.Module,
    path: str | Path,
    size: int = 224,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, size, size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        torch.onnx.export(
            model.eval(),
            (dummy,),
            str(out),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=_OPSET,
            dynamo=False,
        )
    return out
