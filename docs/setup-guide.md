# Setup guide

Public guide for running SafeEyes. This fills in as the pipeline and deployment take shape.

## Development (laptop)

The laptop environment handles dataset processing, model training, and metric computation. It is defined in `pyproject.toml`.

Prerequisites: Python 3.11 or newer (3.12 is used in CI).

Create and activate a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate` instead.

Install the package together with its development extras (pytest, ruff, mypy):

```
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run the quality checks:

```
ruff check .
mypy
pytest
```

These three commands mirror what runs in continuous integration on every push and pull request, so passing them locally means passing CI.

## Edge runtime (Raspberry Pi 4B)

The Pi runs inference only. It loads the quantized ONNX models exported on the
laptop and drives the live loop through ONNX Runtime, with no PyTorch and no
training tooling. The dependency set is `requirements-pi.txt`, intentionally
separate from the laptop environment in `pyproject.toml`.

The model wheels available for a given Pi OS and architecture vary, so treat the
versions in `requirements-pi.txt` as a starting point and pin them against what
actually installs on the target. The steps below are the deployment procedure;
the measured on device latency and frame rate are recorded separately in
[docs/perf/edge-benchmark.md](perf/edge-benchmark.md) once taken on real hardware.

### Prerequisites

- A Raspberry Pi 4B with a 64 bit Pi OS and a Python between 3.10 and 3.12 for
  the runtime environment. MediaPipe's newest wheels for 64 bit ARM stop at
  Python 3.12, so on an OS whose system Python is newer, create the virtual
  environment from a separately installed 3.12 (a standalone build via `uv venv
  --python 3.12` works well and avoids compiling anything on the Pi).
- A cabin facing camera the Pi exposes through V4L2, so OpenCV can open it by
  index.
- The exported int8 ONNX models, produced on the laptop (next step).

### 1. Export the quantized models on the laptop

After training, turn the checkpoints into edge artifacts. This writes both a
float ONNX model and an int8 quantized copy:

```
python -m safeeyes.edge.export_models \
    --temporal-checkpoint models/temporal.pt --n-features 5 \
    --eye-checkpoint models/eye_state.pt \
    --out-dir models/edge
```

The Pi runs from the `.int8.onnx` files.

### 2. Copy the artifacts to the Pi

```
scp models/edge/temporal.int8.onnx models/edge/eye_state.int8.onnx pi@<pi-host>:~/safeeyes-models/
```

### 3. Install the runtime on the Pi

```
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-pi.txt
```

The perception stage uses MediaPipe FaceMesh, and MediaPipe is the dependency
that needs attention on ARM. As of mid 2026 its last release with aarch64
wheels is 0.10.18 (Python 3.10 to 3.12); newer versions install only on x86.
Within that range there is a numpy split: 0.10.15 through 0.10.18 require
numpy below 2, while 0.10.14 accepts modern numpy, so a default resolve lands
on 0.10.14 with numpy 2.x. Both combinations load and run the FaceMesh graph
on a Pi 4B under 64 bit Raspberry Pi OS; pick whichever numpy line the rest of
your environment prefers.

### 4. Launch the live loop

Point the runner at the int8 temporal model and the camera index:

```
python -m safeeyes.edge.run --model ~/safeeyes-models/temporal.int8.onnx --camera 0
```

This opens the camera, runs the perception and temporal stages through ONNX
Runtime, and draws the tiered alert overlay. Press `q` to quit.

### 5. Measure latency and frame rate

Record the on device numbers with the benchmark tool, feeding it the input shape
the runtime uses (batch, window length, feature count for the temporal model;
batch, channel, height, width for the eye state model):

```
python -m safeeyes.edge.bench --model ~/safeeyes-models/temporal.int8.onnx --input-shape 1,150,5
python -m safeeyes.edge.bench --model ~/safeeyes-models/eye_state.int8.onnx --input-shape 1,1,24,24
```

Enter the printed mean, p50, p95, and FPS into the results table in
[docs/perf/edge-benchmark.md](perf/edge-benchmark.md). That table is the single
record of the headline edge numbers, and stays empty until measured here.

### Real time target and the Coral fallback

The goal is a sustained real time frame rate on the Pi's CPU alone. Whether a
Coral USB accelerator is needed is left open until the CPU numbers are measured;
it is the documented fallback if, and only if, the measured frame rate is
insufficient. Adding it is an evidence driven decision, not a default.

### Thermal note

Sustained inference warms the Pi, and thermal throttling lowers the frame rate
over a long run. Note the thermal behavior alongside the latency and FPS numbers
so a short burst figure is not mistaken for sustained performance.

## Datasets

The pipeline trains and evaluates on public datasets. How to obtain them is documented in `docs/source/datasets.md`.
