import json
from pathlib import Path

from safeeyes.data.manifest import read_manifest, write_manifest, write_split
from safeeyes.data.splits import Sample, subject_independent_split


def _samples(n: int) -> list[Sample]:
    return [
        Sample(sample_id=f"s{i:03d}_v", subject_id=f"s{i:03d}", label=["alert", "drowsy"][i % 2])
        for i in range(n)
    ]


def test_manifest_round_trips(tmp_path: Path) -> None:
    samples = _samples(5)
    path = tmp_path / "m.csv"
    write_manifest(samples, path)
    assert read_manifest(path) == samples


def test_manifest_has_a_header_row(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    write_manifest(_samples(2), path)
    first_line = path.read_text().splitlines()[0]
    assert first_line == "sample_id,subject_id,label"


def test_write_split_creates_three_manifests_and_a_summary(tmp_path: Path) -> None:
    split = subject_independent_split(_samples(20), seed=0)
    write_split(split, tmp_path, seed=0)
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "val.csv").exists()
    assert (tmp_path / "test.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_summary_records_seed_and_class_distribution(tmp_path: Path) -> None:
    split = subject_independent_split(_samples(20), seed=3)
    write_split(split, tmp_path, seed=3)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["seed"] == 3
    assert summary["counts"]["train"]["samples"] == len(split.train)
    assert summary["counts"]["train"]["subjects"] == len({s.subject_id for s in split.train})
    assert "class_distribution" in summary["counts"]["train"]
