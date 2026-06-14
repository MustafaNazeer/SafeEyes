import numpy as np
import torch

from safeeyes.alert.pipeline import make_gru_classifier
from safeeyes.edge.export_models import export_and_quantize_eye, export_and_quantize_temporal, main
from safeeyes.edge.runtime import OnnxModel, make_onnx_window_classifier
from safeeyes.models.eye_state import EyeStateCNN
from safeeyes.temporal.model import TemporalGRU


def test_export_and_quantize_temporal_from_checkpoint(tmp_path) -> None:
    torch.manual_seed(7)
    model = TemporalGRU(n_features=5, num_classes=3).eval()
    ckpt = tmp_path / "temporal.pt"
    torch.save(model.state_dict(), ckpt)

    artifacts = export_and_quantize_temporal(ckpt, tmp_path, n_features=5, n_classes=3)

    assert artifacts.onnx.exists()
    assert artifacts.quantized.exists()
    onnx_clf = make_onnx_window_classifier(OnnxModel(artifacts.onnx))
    torch_clf = make_gru_classifier(model)
    rng = np.random.default_rng(0)
    for _ in range(10):
        window = rng.standard_normal((150, 5)).astype(np.float32)
        assert onnx_clf(window) == torch_clf(window)


def test_export_and_quantize_eye_from_checkpoint(tmp_path) -> None:
    model = EyeStateCNN().eval()
    ckpt = tmp_path / "eye.pt"
    torch.save(model.state_dict(), ckpt)
    artifacts = export_and_quantize_eye(ckpt, tmp_path, crop_size=24)
    assert artifacts.onnx.exists()
    assert artifacts.quantized.exists()


def test_export_and_quantize_temporal_can_skip_quantization(tmp_path) -> None:
    model = TemporalGRU(n_features=5, num_classes=3).eval()
    ckpt = tmp_path / "temporal.pt"
    torch.save(model.state_dict(), ckpt)
    artifacts = export_and_quantize_temporal(
        ckpt, tmp_path, n_features=5, n_classes=3, quantize=False
    )
    assert artifacts.onnx.exists()
    assert artifacts.quantized is None


def test_main_exports_temporal_checkpoint(tmp_path) -> None:
    model = TemporalGRU(n_features=5, num_classes=3).eval()
    ckpt = tmp_path / "temporal.pt"
    torch.save(model.state_dict(), ckpt)
    out_dir = tmp_path / "artifacts"
    code = main(
        ["--temporal-checkpoint", str(ckpt), "--out-dir", str(out_dir), "--n-features", "5"]
    )
    assert code == 0
    assert (out_dir / "temporal.onnx").exists()
    assert (out_dir / "temporal.int8.onnx").exists()
