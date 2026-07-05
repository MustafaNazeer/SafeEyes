import io
import json

from safeeyes.observability.emitter import JsonlEmitter


def test_emit_writes_one_json_object_per_call() -> None:
    stream = io.StringIO()
    emitter = JsonlEmitter(stream=stream, clock=lambda: 1000.0)

    emitter.emit("start", window=150)
    emitter.emit("stop")

    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {"ts": 1000.0, "event": "start", "window": 150}
    assert json.loads(lines[1]) == {"ts": 1000.0, "event": "stop"}


def test_timestamp_comes_from_the_clock() -> None:
    stream = io.StringIO()
    ticks = iter([1.5, 2.5])
    emitter = JsonlEmitter(stream=stream, clock=lambda: next(ticks))

    emitter.emit("a")
    emitter.emit("b")

    tss = [json.loads(line)["ts"] for line in stream.getvalue().splitlines()]
    assert tss == [1.5, 2.5]


def test_arbitrary_scalar_fields_are_merged() -> None:
    stream = io.StringIO()
    emitter = JsonlEmitter(stream=stream, clock=lambda: 0.0)

    emitter.emit("metrics", fps=21.3, face_rate=0.94, frames=107)

    record = json.loads(stream.getvalue().strip())
    assert record["fps"] == 21.3
    assert record["face_rate"] == 0.94
    assert record["frames"] == 107
