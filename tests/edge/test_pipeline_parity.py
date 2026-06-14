import numpy as np
import pytest
import torch

from safeeyes.alert.pipeline import DrowsinessPipeline, make_gru_classifier
from safeeyes.edge.export import export_temporal_onnx
from safeeyes.edge.runtime import OnnxModel, make_onnx_window_classifier
from safeeyes.temporal.model import TemporalGRU


def _tier_sequence(classifier, feature_stream, window_capacity: int, n_features: int):
    pipeline = DrowsinessPipeline(
        classifier=classifier, window_capacity=window_capacity, n_features=n_features
    )
    return [pipeline.process(features) for features in feature_stream]


@pytest.mark.parametrize("model_seed", [11, 23, 99])
def test_torch_and_onnx_pipelines_produce_identical_tier_sequences(tmp_path, model_seed) -> None:
    torch.manual_seed(model_seed)
    model = TemporalGRU(n_features=5, num_classes=3).eval()
    path = export_temporal_onnx(model, tmp_path / f"t{model_seed}.onnx", n_features=5)

    rng = np.random.default_rng(model_seed)
    feature_stream = [rng.standard_normal(5).astype(np.float32) for _ in range(200)]

    torch_tiers = _tier_sequence(make_gru_classifier(model), feature_stream, 20, 5)
    onnx_tiers = _tier_sequence(
        make_onnx_window_classifier(OnnxModel(path)), feature_stream, 20, 5
    )

    assert onnx_tiers == torch_tiers
