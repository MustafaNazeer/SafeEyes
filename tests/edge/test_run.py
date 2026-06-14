import numpy as np

from safeeyes.alert.pipeline import DrowsinessPipeline
from safeeyes.alert.state_machine import AlertTier
from safeeyes.edge.export import export_temporal_onnx
from safeeyes.edge.run import build_onnx_classifier
from safeeyes.temporal.model import TemporalGRU


def test_build_onnx_classifier_drives_pipeline_from_file(tmp_path) -> None:
    model = TemporalGRU(n_features=5, num_classes=3).eval()
    path = export_temporal_onnx(model, tmp_path / "t.onnx", n_features=5)

    classifier = build_onnx_classifier(path)
    pipeline = DrowsinessPipeline(classifier=classifier, window_capacity=10, n_features=5)
    tier = AlertTier.NONE
    for _ in range(12):
        tier = pipeline.process(np.zeros(5, dtype=np.float32))
    assert isinstance(tier, AlertTier)


def test_build_onnx_classifier_returns_class_index(tmp_path) -> None:
    model = TemporalGRU(n_features=5, num_classes=3).eval()
    path = export_temporal_onnx(model, tmp_path / "t.onnx", n_features=5)

    classifier = build_onnx_classifier(path)
    level = classifier(np.zeros((10, 5), dtype=np.float32))
    assert isinstance(level, int)
    assert 0 <= level < 3
