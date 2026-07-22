import numpy as np

from safeeyes.alert.pipeline import DrowsinessPipeline
from safeeyes.alert.state_machine import AlertTier
from safeeyes.distraction.labels import DISTRACTION_LABELS
from safeeyes.distraction.scheduler import DistractionScheduler
from safeeyes.edge import run as run_module
from safeeyes.edge.export import export_distraction_onnx, export_temporal_onnx
from safeeyes.edge.run import build_distraction_scheduler, build_onnx_classifier, main
from safeeyes.models.train_distraction import BACKBONES
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


def test_build_distraction_scheduler_runs_from_file(tmp_path) -> None:
    model = BACKBONES["mobilenet_v3_small"](len(DISTRACTION_LABELS), False).eval()
    path = export_distraction_onnx(model, tmp_path / "d.onnx", size=64)

    scheduler = build_distraction_scheduler(path, every_n=1, size=64)
    assert isinstance(scheduler, DistractionScheduler)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    state = scheduler.update(frame)
    assert state.ran is True
    assert state.activity in DISTRACTION_LABELS


def test_main_wires_logging_arguments_into_run(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(run_module, "run", lambda **kwargs: calls.update(kwargs))

    exit_code = main(
        [
            "--model",
            "models/edge/temporal.int8.onnx",
            "--camera",
            "1",
            "--window",
            "120",
            "--log-file",
            "edge.jsonl",
            "--metrics-interval",
            "10",
        ]
    )

    assert exit_code == 0
    assert calls == {
        "model_path": "models/edge/temporal.int8.onnx",
        "camera_index": 1,
        "window_capacity": 120,
        "distraction_model": None,
        "distraction_every_n": 5,
        "distraction_alpha": 0.5,
        "gaze_model": None,
        "gaze_min_seconds": 2.0,
        "gaze_audible_seconds": 4.0,
        "gaze_smoothing_seconds": 1.0,
        "log_file": "edge.jsonl",
        "metrics_interval": 10.0,
        "show_display": True,
    }


def test_main_wires_distraction_arguments_into_run(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(run_module, "run", lambda **kwargs: calls.update(kwargs))

    exit_code = main(
        [
            "--model",
            "models/edge/temporal.int8.onnx",
            "--distraction-model",
            "models/edge/distraction.onnx",
            "--distraction-every-n",
            "3",
            "--distraction-alpha",
            "0.25",
        ]
    )

    assert exit_code == 0
    assert calls["distraction_model"] == "models/edge/distraction.onnx"
    assert calls["distraction_every_n"] == 3
    assert calls["distraction_alpha"] == 0.25


def test_main_wires_no_display_into_run(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(run_module, "run", lambda **kwargs: calls.update(kwargs))

    exit_code = main(["--model", "models/edge/temporal.int8.onnx", "--no-display"])

    assert exit_code == 0
    assert calls["show_display"] is False


def test_main_wires_gaze_arguments_into_run(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(run_module, "run", lambda **kwargs: calls.update(kwargs))

    exit_code = main(
        [
            "--model",
            "models/edge/temporal.int8.onnx",
            "--gaze-model",
            "models/gaze/edge/gaze_zone.onnx",
            "--gaze-min-seconds",
            "1.5",
            "--gaze-audible-seconds",
            "3.0",
            "--gaze-smoothing-seconds",
            "0.5",
        ]
    )

    assert exit_code == 0
    assert calls["gaze_model"] == "models/gaze/edge/gaze_zone.onnx"
    assert calls["gaze_min_seconds"] == 1.5
    assert calls["gaze_audible_seconds"] == 3.0
    assert calls["gaze_smoothing_seconds"] == 0.5
