import numpy as np
import torch

from safeeyes.alert.pipeline import make_gru_classifier
from safeeyes.edge.export_models import (
    export_and_quantize_distraction,
    export_and_quantize_eye,
    export_and_quantize_temporal,
    main,
)
from safeeyes.edge.runtime import OnnxModel, make_onnx_window_classifier
from safeeyes.models.distraction_data import DISTRACTION_LABELS
from safeeyes.models.eye_state import EyeStateCNN
from safeeyes.models.train_distraction import BACKBONES
from safeeyes.perception.frame import FEATURE_COLUMNS
from safeeyes.temporal.model import TemporalGRU


def _save_distraction_checkpoint(path, backbone: str) -> torch.nn.Module:
    labels = list(DISTRACTION_LABELS)
    model = BACKBONES[backbone](len(labels), False).eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": backbone,
            "labels": labels,
            "num_classes": len(labels),
        },
        path,
    )
    return model


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


def test_export_and_quantize_distraction_from_checkpoint(tmp_path) -> None:
    backbone = "mobilenet_v3_small"
    ckpt = tmp_path / "distraction.pt"
    model = _save_distraction_checkpoint(ckpt, backbone)

    artifacts = export_and_quantize_distraction(ckpt, tmp_path, quantize=False)

    assert artifacts.onnx.name == f"distraction_{backbone}.onnx"
    assert artifacts.onnx.exists()
    assert artifacts.quantized is None

    x = np.random.randn(2, 3, 224, 224).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    actual = OnnxModel(artifacts.onnx).run(x)
    assert actual.shape == (2, len(DISTRACTION_LABELS))
    np.testing.assert_allclose(actual, expected, atol=1e-3)


def test_export_and_quantize_distraction_writes_int8_companion(tmp_path) -> None:
    backbone = "shufflenet_v2_x0_5"
    ckpt = tmp_path / "distraction.pt"
    _save_distraction_checkpoint(ckpt, backbone)

    artifacts = export_and_quantize_distraction(ckpt, tmp_path, quantize=True)

    assert artifacts.onnx.name == f"distraction_{backbone}.onnx"
    assert artifacts.quantized is not None
    assert artifacts.quantized.name == f"distraction_{backbone}.int8.onnx"
    assert artifacts.quantized.exists()


def test_distraction_int8_onnx_loads_and_runs(tmp_path) -> None:
    # Float parity is asserted above; int8 is dynamic-quantized (approximate), so
    # we only require that it loads and produces the right output shape, matching
    # the v1 policy of not asserting exact int8 parity.
    backbone = "mobilenet_v3_small"
    ckpt = tmp_path / "distraction.pt"
    _save_distraction_checkpoint(ckpt, backbone)

    artifacts = export_and_quantize_distraction(ckpt, tmp_path, quantize=True)

    assert artifacts.quantized is not None
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    out = OnnxModel(artifacts.quantized).run(x)
    assert out.shape == (1, len(DISTRACTION_LABELS))


def test_main_exports_distraction_checkpoint(tmp_path) -> None:
    backbone = "mobilenet_v3_small"
    ckpt = tmp_path / "distraction.pt"
    _save_distraction_checkpoint(ckpt, backbone)
    out_dir = tmp_path / "artifacts"
    code = main(
        ["--distraction-checkpoint", str(ckpt), "--out-dir", str(out_dir)]
    )
    assert code == 0
    assert (out_dir / f"distraction_{backbone}.onnx").exists()
    assert (out_dir / f"distraction_{backbone}.int8.onnx").exists()


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


def test_main_temporal_default_matches_the_runtime_feature_count(tmp_path) -> None:
    """An export run without --n-features must load in the live loop.

    Both live loops build the window from len(FEATURE_COLUMNS), so a default
    that disagrees with it yields an artifact the runtime cannot consume.
    """
    model = TemporalGRU(n_features=len(FEATURE_COLUMNS), num_classes=3).eval()
    ckpt = tmp_path / "temporal.pt"
    torch.save(model.state_dict(), ckpt)
    out_dir = tmp_path / "artifacts"

    code = main(["--temporal-checkpoint", str(ckpt), "--out-dir", str(out_dir)])

    assert code == 0
    classifier = make_onnx_window_classifier(OnnxModel(out_dir / "temporal.onnx"))
    window = np.zeros((150, len(FEATURE_COLUMNS)), dtype=np.float32)
    assert classifier(window) in {0, 1, 2}
