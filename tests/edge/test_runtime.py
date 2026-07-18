import numpy as np
import torch

from safeeyes.alert.pipeline import DrowsinessPipeline, make_gru_classifier
from safeeyes.alert.state_machine import AlertTier
from safeeyes.edge.export import (
    export_distraction_onnx,
    export_eye_state_onnx,
    export_temporal_onnx,
)
from safeeyes.edge.runtime import (
    OnnxModel,
    make_onnx_distraction_classifier,
    make_onnx_eye_classifier,
    make_onnx_window_classifier,
)
from safeeyes.models.distraction_data import DISTRACTION_LABELS
from safeeyes.models.eye_state import EyeStateCNN
from safeeyes.models.train_distraction import BACKBONES
from safeeyes.temporal.model import TemporalGRU


def test_onnx_model_run_returns_logits(tmp_path) -> None:
    torch_model = TemporalGRU(n_features=5, num_classes=3).eval()
    path = export_temporal_onnx(torch_model, tmp_path / "t.onnx", n_features=5)
    model = OnnxModel(path)
    out = model.run(np.zeros((1, 150, 5), dtype=np.float32))
    assert out.shape == (1, 3)


def test_onnx_window_classifier_matches_torch_decisions(tmp_path) -> None:
    torch.manual_seed(3)
    torch_model = TemporalGRU(n_features=5, num_classes=3).eval()
    path = export_temporal_onnx(torch_model, tmp_path / "t.onnx", n_features=5)
    onnx_clf = make_onnx_window_classifier(OnnxModel(path))
    torch_clf = make_gru_classifier(torch_model)
    rng = np.random.default_rng(0)
    for _ in range(15):
        window = rng.standard_normal((150, 5)).astype(np.float32)
        assert onnx_clf(window) == torch_clf(window)


def test_onnx_window_classifier_drops_into_pipeline(tmp_path) -> None:
    torch_model = TemporalGRU(n_features=5, num_classes=3).eval()
    path = export_temporal_onnx(torch_model, tmp_path / "t.onnx", n_features=5)
    onnx_clf = make_onnx_window_classifier(OnnxModel(path))
    pipeline = DrowsinessPipeline(classifier=onnx_clf, window_capacity=10, n_features=5)
    tier = AlertTier.NONE
    for _ in range(12):
        tier = pipeline.process(np.zeros(5, dtype=np.float32))
    assert isinstance(tier, AlertTier)


def test_onnx_distraction_classifier_matches_torch_softmax(tmp_path) -> None:
    torch.manual_seed(5)
    num_classes = len(DISTRACTION_LABELS)
    torch_model = BACKBONES["mobilenet_v3_small"](num_classes, False).eval()
    path = export_distraction_onnx(torch_model, tmp_path / "d.onnx", size=64)
    onnx_clf = make_onnx_distraction_classifier(OnnxModel(path))

    rng = np.random.default_rng(2)
    image = rng.standard_normal((1, 3, 64, 64)).astype(np.float32)
    probs = onnx_clf(image)

    with torch.no_grad():
        expected = torch.softmax(torch_model(torch.from_numpy(image)), dim=1).numpy()[0]

    assert probs.shape == (num_classes,)
    np.testing.assert_allclose(float(probs.sum()), 1.0, atol=1e-5)
    np.testing.assert_allclose(probs, expected, atol=1e-4)
    assert int(probs.argmax()) == int(expected.argmax())


def test_onnx_eye_classifier_matches_torch_decisions(tmp_path) -> None:
    torch.manual_seed(4)
    torch_model = EyeStateCNN().eval()
    path = export_eye_state_onnx(torch_model, tmp_path / "eye.onnx", crop_size=24)
    onnx_eye = make_onnx_eye_classifier(OnnxModel(path))
    rng = np.random.default_rng(1)
    for _ in range(15):
        crop = rng.standard_normal((24, 24)).astype(np.float32)
        with torch.no_grad():
            expected = int(torch_model(torch.from_numpy(crop)[None, None]).argmax(dim=1).item())
        assert onnx_eye(crop) == expected
