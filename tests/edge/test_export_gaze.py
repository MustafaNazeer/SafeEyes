import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from safeeyes.edge.runtime import OnnxModel


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
