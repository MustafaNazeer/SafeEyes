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

Deployment steps will be documented here: preparing the Pi, installing the runtime, loading the quantized models, connecting the camera, and launching the real time loop. The Pi dependency set is the lightweight inference environment defined in `requirements-pi.txt`, which is intentionally separate from the laptop development environment in `pyproject.toml`.

## Datasets

The pipeline trains and evaluates on public datasets. How to obtain them is documented in `docs/source/datasets.md`.
