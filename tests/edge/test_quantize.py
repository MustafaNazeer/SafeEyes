import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

from safeeyes.edge.export import export_eye_state_onnx, export_temporal_onnx
from safeeyes.edge.quantize import quantize_dynamic_onnx
from safeeyes.models.eye_state import EyeStateCNN
from safeeyes.temporal.model import TemporalGRU


def _has_int8_weights(path: str) -> bool:
    model = onnx.load(path)
    return any(init.data_type == TensorProto.INT8 for init in model.graph.initializer)


def _run(path: str, array: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return np.asarray(session.run(None, {name: array})[0])


def test_quantize_writes_file(tmp_path) -> None:
    src = export_eye_state_onnx(EyeStateCNN().eval(), tmp_path / "eye.onnx")
    dst = quantize_dynamic_onnx(src, tmp_path / "eye.int8.onnx")
    assert dst.exists()


def test_quantized_eye_graph_contains_int8_weights(tmp_path) -> None:
    src = export_eye_state_onnx(EyeStateCNN().eval(), tmp_path / "eye.onnx")
    dst = quantize_dynamic_onnx(src, tmp_path / "eye.int8.onnx")
    assert _has_int8_weights(str(dst))


def test_quantized_eye_runs_with_correct_shape(tmp_path) -> None:
    src = export_eye_state_onnx(EyeStateCNN().eval(), tmp_path / "eye.onnx")
    dst = quantize_dynamic_onnx(src, tmp_path / "eye.int8.onnx")
    out = _run(str(dst), np.random.randn(4, 1, 24, 24).astype(np.float32))
    assert out.shape == (4, 2)
    assert np.isfinite(out).all()


def test_quantized_temporal_graph_contains_int8_weights(tmp_path) -> None:
    torch_model = TemporalGRU(n_features=5, num_classes=3).eval()
    src = export_temporal_onnx(torch_model, tmp_path / "t.onnx", n_features=5)
    dst = quantize_dynamic_onnx(src, tmp_path / "t.int8.onnx")
    assert _has_int8_weights(str(dst))


def test_quantized_temporal_runs_with_correct_shape(tmp_path) -> None:
    torch_model = TemporalGRU(n_features=5, num_classes=3).eval()
    src = export_temporal_onnx(torch_model, tmp_path / "t.onnx", n_features=5)
    dst = quantize_dynamic_onnx(src, tmp_path / "t.int8.onnx")
    out = _run(str(dst), np.random.randn(2, 150, 5).astype(np.float32))
    assert out.shape == (2, 3)
    assert np.isfinite(out).all()
