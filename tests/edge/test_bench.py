from safeeyes.edge.bench import main, run_benchmark
from safeeyes.edge.benchmark import BenchmarkResult
from safeeyes.edge.export import export_temporal_onnx
from safeeyes.temporal.model import TemporalGRU


def test_run_benchmark_returns_result(tmp_path) -> None:
    torch_model = TemporalGRU(n_features=5, num_classes=3).eval()
    path = export_temporal_onnx(torch_model, tmp_path / "t.onnx", n_features=5)
    result = run_benchmark(path, (1, 150, 5), runs=8, warmup=2)
    assert isinstance(result, BenchmarkResult)
    assert result.runs == 8


def test_main_prints_latency_report(tmp_path, capsys) -> None:
    torch_model = TemporalGRU(n_features=5, num_classes=3).eval()
    path = export_temporal_onnx(torch_model, tmp_path / "t.onnx", n_features=5)
    code = main(["--model", str(path), "--input-shape", "1,150,5", "--runs", "5", "--warmup", "1"])
    assert code == 0
    out = capsys.readouterr().out.lower()
    assert "fps" in out
    assert "ms" in out
