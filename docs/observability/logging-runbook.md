# Logging and metrics runbook

Public note on the structured telemetry the live demo emits, how to turn it on,
and how to read it. Both runners (the laptop PyTorch loop and the Raspberry Pi
ONNX loop) share the same instrumentation, so the log schema is identical on
either device.

The telemetry is scalar only by design: alert transitions, face-detection edges,
and periodic performance summaries. No video frame, image crop, or landmark
coordinate is ever written. That boundary is enforced by a static test over the
source tree, so a future change cannot quietly start persisting raw video or
reaching the network. See the privacy threat model for the full posture.

## Turning it on

Both runners log to standard error by default, one JSON object per line. Point
them at a file with `--log-file`, and set how often the rolling performance
summary is emitted with `--metrics-interval` (seconds, default 5):

```bash
# laptop
python -m safeeyes.alert.run --checkpoint models/temporal.pt \
    --log-file run.jsonl --metrics-interval 5

# Raspberry Pi
python -m safeeyes.edge.run --model models/edge/temporal.int8.onnx \
    --log-file run.jsonl --metrics-interval 5
```

Because every line is a self-contained JSON object, the file is easy to filter
with standard tools, for example `grep '"event":"tier_change"' run.jsonl` or a
one-line `jq` selection.

## Event schema

Every record carries a `ts` (wall-clock seconds) and an `event` name. The
remaining fields depend on the event.

| Event | When | Fields |
|-------|------|--------|
| `start` | Once at loop startup | `config`: the run parameters (backend, model name, camera index, window size, and the alert-tier thresholds) |
| `tier_change` | The alert tier changes | `from`, `to` (tier names), `fatigue` (the classifier level at the transition) |
| `face_lost` | A face was being tracked and is no longer detected | none |
| `face_regained` | A face is detected again after being lost | none |
| `metrics` | Every `--metrics-interval` seconds, and once at shutdown | `frames`, `fps`, `lat_ms_p50`, `lat_ms_p95`, `face_rate` for the window |
| `stop` | Once at loop shutdown | none |

Face-detection events are edge triggered: they fire on the transition, not on
every frame, so a driver looking away produces one `face_lost` and later one
`face_regained` rather than a burst.

## Reading the metrics line

The `metrics` record summarizes the window since the previous one:

- `frames`: frames processed in the window.
- `fps`: real throughput, frames divided by the wall-clock span of the window.
  This is end-to-end loop rate, not raw model inference speed; the isolated
  inference latency lives in the edge benchmark note.
- `lat_ms_p50`, `lat_ms_p95`: median and 95th percentile per-frame processing
  latency in milliseconds, covering landmark detection through the alert
  decision.
- `face_rate`: fraction of frames in the window in which a face was detected.

A healthy run shows a steady `fps`, a `lat_ms_p95` within the per-frame budget,
and a `face_rate` near 1.0 while the subject faces the camera.

## Retention on a long session

The runtime never rotates the log itself; it appends. For an unattended session
on the Pi, keep the file bounded with the platform's own tooling, for example a
`logrotate` rule on the log path or piping standard error through `rotatelogs`.
The records are small and fixed in shape, so a size-based rotation is sufficient;
there is no structured index to preserve across a rotation.
