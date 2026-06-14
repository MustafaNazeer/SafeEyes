"""Benchmark an ONNX model from the command line.

This is the tool that produces the on-device latency and frame rate numbers. Copy
a ``.int8.onnx`` artifact to the Pi, run this against it with the input shape the
runtime feeds, and record the printed mean, p50, p95, and FPS in the perf notes.
A zero-filled sample of the given shape is enough for a timing measurement; the
numbers depend on the graph and the hardware, not the input values.

    python -m safeeyes.edge.bench --model temporal.int8.onnx --input-shape 1,150,8
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from safeeyes.edge.benchmark import BenchmarkResult, benchmark_model
from safeeyes.edge.runtime import OnnxModel


def run_benchmark(
    model_path: str | Path,
    input_shape: tuple[int, ...],
    runs: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    model = OnnxModel(model_path)
    sample = np.zeros(input_shape, dtype=np.float32)
    return benchmark_model(model, sample, runs=runs, warmup=warmup)


def _parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(","))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark an ONNX model's inference latency.")
    parser.add_argument("--model", required=True, help="path to the .onnx or .int8.onnx model")
    parser.add_argument(
        "--input-shape", required=True, help="comma separated input shape, e.g. 1,150,8"
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args(argv)

    result = run_benchmark(args.model, _parse_shape(args.input_shape), args.runs, args.warmup)
    print(
        f"runs={result.runs} "
        f"mean={result.mean_ms:.3f}ms "
        f"p50={result.p50_ms:.3f}ms "
        f"p95={result.p95_ms:.3f}ms "
        f"throughput={result.fps:.1f}fps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
