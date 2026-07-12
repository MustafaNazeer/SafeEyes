import numpy as np

from safeeyes.alert.pipeline import DrowsinessPipeline
from safeeyes.alert.state_machine import AlertTier
from safeeyes.edge import run as run_module
from safeeyes.edge.export import export_temporal_onnx
from safeeyes.edge.run import build_onnx_classifier, main
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


def test_main_wires_logging_arguments_into_run(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(run_module, "run", lambda **kwargs: calls.update(kwargs))

    exit_code = main(
        [
            "--model", "models/edge/temporal.int8.onnx",
            "--camera", "1",
            "--window", "120",
            "--log-file", "edge.jsonl",
            "--metrics-interval", "10",
        ]
    )

    assert exit_code == 0
    assert calls == {
        "model_path": "models/edge/temporal.int8.onnx",
        "camera_index": 1,
        "window_capacity": 120,
        "log_file": "edge.jsonl",
        "metrics_interval": 10.0,
        "show_display": True,
    }


def test_main_wires_no_display_into_run(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(run_module, "run", lambda **kwargs: calls.update(kwargs))

    exit_code = main(["--model", "models/edge/temporal.int8.onnx", "--no-display"])

    assert exit_code == 0
    assert calls["show_display"] is False
