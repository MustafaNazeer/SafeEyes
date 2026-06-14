import numpy as np
import onnxruntime as ort
import torch

from safeeyes.edge.export import export_eye_state_onnx, export_temporal_onnx
from safeeyes.models.eye_state import EyeStateCNN
from safeeyes.temporal.model import TemporalGRU


def _run_onnx(path: str, array: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return np.asarray(session.run(None, {name: array})[0])


def test_eye_state_export_writes_file(tmp_path) -> None:
    model = EyeStateCNN()
    out = export_eye_state_onnx(model, tmp_path / "eye.onnx")
    assert out.exists()
    assert out.suffix == ".onnx"


def test_eye_state_onnx_matches_torch(tmp_path) -> None:
    torch.manual_seed(0)
    model = EyeStateCNN().eval()
    out = export_eye_state_onnx(model, tmp_path / "eye.onnx", crop_size=24)
    x = torch.randn(1, 1, 24, 24)
    with torch.no_grad():
        expected = model(x).numpy()
    actual = _run_onnx(str(out), x.numpy())
    assert actual.shape == (1, 2)
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_eye_state_onnx_accepts_dynamic_batch(tmp_path) -> None:
    model = EyeStateCNN().eval()
    out = export_eye_state_onnx(model, tmp_path / "eye.onnx", crop_size=24)
    batch = np.random.randn(5, 1, 24, 24).astype(np.float32)
    assert _run_onnx(str(out), batch).shape == (5, 2)


def test_temporal_export_writes_file(tmp_path) -> None:
    model = TemporalGRU(n_features=5, num_classes=3)
    out = export_temporal_onnx(model, tmp_path / "temporal.onnx", n_features=5)
    assert out.exists()


def test_temporal_onnx_matches_torch(tmp_path) -> None:
    torch.manual_seed(1)
    model = TemporalGRU(n_features=5, num_classes=3).eval()
    out = export_temporal_onnx(model, tmp_path / "temporal.onnx", n_features=5, seq_len=150)
    x = torch.randn(1, 150, 5)
    with torch.no_grad():
        expected = model(x).numpy()
    actual = _run_onnx(str(out), x.numpy())
    assert actual.shape == (1, 3)
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_temporal_onnx_accepts_dynamic_sequence_length(tmp_path) -> None:
    model = TemporalGRU(n_features=5, num_classes=3).eval()
    out = export_temporal_onnx(model, tmp_path / "temporal.onnx", n_features=5, seq_len=150)
    shorter = np.random.randn(2, 90, 5).astype(np.float32)
    assert _run_onnx(str(out), shorter).shape == (2, 3)
