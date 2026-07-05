"""Seam tests for the live demo runner.

The camera loop is exercised live, but the two seams it orchestrates are unit
tested here: rebuilding the temporal model from a checkpoint, and the argument
wiring from the command line into ``run``.
"""

import pytest
import torch

from safeeyes.alert import run as run_module
from safeeyes.alert.run import _build_model, main
from safeeyes.temporal.model import TemporalGRU


def test_build_model_round_trips_a_checkpoint(tmp_path) -> None:
    original = TemporalGRU(n_features=5, num_classes=3).eval()
    checkpoint = tmp_path / "temporal.pt"
    torch.save(original.state_dict(), checkpoint)

    restored = _build_model(str(checkpoint), n_features=5, n_classes=3).eval()

    sequence = torch.zeros(1, 12, 5)
    with torch.no_grad():
        assert torch.allclose(original(sequence), restored(sequence), atol=1e-6)


def test_main_wires_arguments_into_run(monkeypatch) -> None:
    calls = {}

    def fake_run(**kwargs) -> None:
        calls.update(kwargs)

    monkeypatch.setattr(run_module, "run", fake_run)
    exit_code = main(
        [
            "--checkpoint", "models/temporal.pt",
            "--camera", "2",
            "--window", "99",
            "--log-file", "run.jsonl",
            "--metrics-interval", "2.5",
        ]
    )

    assert exit_code == 0
    assert calls == {
        "checkpoint": "models/temporal.pt",
        "camera_index": 2,
        "window_capacity": 99,
        "log_file": "run.jsonl",
        "metrics_interval": 2.5,
    }


def test_main_logging_arguments_default_to_stderr_and_five_seconds(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(run_module, "run", lambda **kwargs: calls.update(kwargs))

    main(["--checkpoint", "models/temporal.pt"])

    assert calls["log_file"] is None
    assert calls["metrics_interval"] == 5.0


def test_main_requires_a_checkpoint() -> None:
    with pytest.raises(SystemExit):
        main([])
