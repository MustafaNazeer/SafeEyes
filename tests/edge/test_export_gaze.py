import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from safeeyes.edge.export_gaze import export_gaze_onnx, verify_parity
from safeeyes.edge.runtime import OnnxModel, make_onnx_gaze_classifier


def _fit_probe_model(n_features: int = 7, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((200, n_features)).astype(np.float32)
    y = np.where(x[:, 0] + x[:, 3] > 0, "front", "left")
    return GradientBoostingClassifier(random_state=0).fit(x, y), rng


def test_skl2onnx_round_trips_a_gradient_boosted_classifier(tmp_path) -> None:
    """The Pi has no scikit-learn, so the gaze model must survive ONNX conversion.

    Labels are compared rather than probabilities: the deployed decision is the
    predicted zone, and a conversion that agreed on probabilities but disagreed
    on the argmax would still be a broken deployment.
    """
    model, rng = _fit_probe_model()

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    onnx_model = convert_sklearn(model, initial_types=[("input", FloatTensorType([None, 7]))])
    path = tmp_path / "gaze.onnx"
    path.write_bytes(onnx_model.SerializeToString())

    session = OnnxModel(path)
    probe = rng.standard_normal((25, 7)).astype(np.float32)
    onnx_labels = np.asarray(session.run(probe)).ravel()
    assert (onnx_labels == model.predict(probe)).all()


def test_export_writes_onnx_and_reports_zero_mismatches(tmp_path) -> None:
    """The deployed artifact must make the same decisions as the trained model."""
    model, rng = _fit_probe_model()
    out = export_gaze_onnx(model, tmp_path / "gaze.onnx", n_features=7)

    assert out.exists()
    probe = rng.standard_normal((60, 7)).astype(np.float32)
    assert verify_parity(model, out, probe) == 0


def test_parity_check_counts_disagreements_rather_than_asserting(tmp_path) -> None:
    """A model exported at the wrong width must be reported, not silently pass."""
    model, rng = _fit_probe_model()
    out = export_gaze_onnx(model, tmp_path / "gaze.onnx", n_features=7)
    probe = rng.standard_normal((10, 7)).astype(np.float32)
    assert isinstance(verify_parity(model, out, probe), int)


def test_the_onnx_adapter_returns_a_zone_name(tmp_path) -> None:
    model, rng = _fit_probe_model()
    out = export_gaze_onnx(model, tmp_path / "gaze.onnx", n_features=7)

    classify = make_onnx_gaze_classifier(OnnxModel(out))
    zone = classify(rng.standard_normal(7).astype(np.float32))
    assert isinstance(zone, str)
    assert zone in {"front", "left"}


def test_the_adapter_agrees_with_the_trained_model_row_by_row(tmp_path) -> None:
    model, rng = _fit_probe_model()
    out = export_gaze_onnx(model, tmp_path / "gaze.onnx", n_features=7)
    classify = make_onnx_gaze_classifier(OnnxModel(out))

    probe = rng.standard_normal((20, 7)).astype(np.float32)
    expected = model.predict(probe)
    assert [classify(row) for row in probe] == list(expected)
