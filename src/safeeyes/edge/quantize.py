"""Quantize an ONNX model to int8 for the edge runtime.

Dynamic quantization is used: weights are stored as int8 and activations are
quantized on the fly at inference, which needs no calibration dataset and suits
the small models here. Before quantizing, the intermediate shape annotations are
cleared and re-inferred, because the graph-capture exporter leaves shape
metadata that the quantizer's own shape inference rejects.

Size is not the point for these compact models; on a tiny graph the per-tensor
quantization overhead can outweigh the int8 weight savings. The point is an int8
weight path that runs under the same ONNX Runtime the Pi uses, so the on-device
benchmark measures the quantized model rather than the float one.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_dynamic_onnx(src: str | Path, dst: str | Path) -> Path:
    src_path = Path(src)
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(src_path))
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    prepared = out.with_suffix(".prepared.onnx")
    onnx.save(model, str(prepared))

    try:
        quantize_dynamic(str(prepared), str(out), weight_type=QuantType.QInt8)
    finally:
        prepared.unlink(missing_ok=True)
    return out
