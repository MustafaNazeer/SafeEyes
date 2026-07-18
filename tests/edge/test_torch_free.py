"""The edge runtime must be importable without PyTorch installed.

The Raspberry Pi environment (requirements-pi.txt) deliberately excludes torch;
the live loop there runs entirely through ONNX Runtime. A module level torch
import anywhere in the edge runner's import chain breaks that promise, and it
only surfaces on hardware because every development machine has torch. This
guard blocks torch in a subprocess and imports the edge entry point.
"""

import subprocess
import sys

BLOCK_TORCH = "import sys; sys.modules['torch'] = None; "


def _import_in_subprocess(statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", BLOCK_TORCH + statement + "; print('imported')"],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_edge_run_imports_without_torch() -> None:
    result = _import_in_subprocess("import safeeyes.edge.run")
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


def test_pipeline_core_imports_without_torch() -> None:
    result = _import_in_subprocess("import safeeyes.alert.pipeline")
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


def test_distraction_preprocess_imports_without_torch() -> None:
    result = _import_in_subprocess("import safeeyes.edge.preprocess")
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


def test_torch_block_actually_blocks() -> None:
    # Positive control: the guard would be vacuous if the block did not work.
    result = _import_in_subprocess("import torch")
    assert result.returncode != 0
