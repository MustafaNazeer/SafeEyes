import json

from safeeyes.observability.session import make_run_observer


def test_make_run_observer_writes_events_to_a_log_file(tmp_path) -> None:
    log = tmp_path / "run.jsonl"
    observer, closer = make_run_observer(log_file=str(log), metrics_interval_s=5.0)

    observer.start({"backend": "torch", "window": 150})
    observer.observe(tier="none", fatigue=0, face_detected=True, latency_s=0.01)
    assert closer is not None
    closer.close()

    lines = log.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["event"] == "start"
    assert json.loads(lines[0])["config"] == {"backend": "torch", "window": 150}


def test_make_run_observer_defaults_to_stderr(capsys) -> None:
    observer, closer = make_run_observer(log_file=None)

    assert closer is None
    observer.start({})

    err = capsys.readouterr().err
    assert json.loads(err.strip())["event"] == "start"
