from __future__ import annotations

from pathlib import Path

import pytest

from safeeyes.data.intervals import (
    IntervalSample,
    read_interval_manifest,
    write_interval_manifest,
    write_interval_split,
)
from safeeyes.data.splits import (
    Split,
    SubjectLeakageError,
    assert_subject_independent,
    subject_independent_split,
)


def _sample(i: int, subject: str = "gA_1", label: str = "drinking") -> IntervalSample:
    return IntervalSample(
        sample_id=f"gA/1/s1/video.mp4#{i * 100}-{i * 100 + 99}",
        subject_id=subject,
        label=label,
        start_frame=i * 100,
        end_frame=i * 100 + 99,
    )


def test_interval_manifest_round_trips(tmp_path: Path) -> None:
    samples = [_sample(0), _sample(1, subject="gB_6", label="radio")]
    path = tmp_path / "manifest.csv"
    write_interval_manifest(samples, path)
    assert read_interval_manifest(path) == samples


def test_interval_manifest_header_carries_frame_columns(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    write_interval_manifest([_sample(0)], path)
    header = path.read_text().splitlines()[0]
    assert header == "sample_id,subject_id,label,start_frame,end_frame"


def test_interval_samples_split_subject_independently() -> None:
    samples = [
        _sample(i, subject=f"g{chr(65 + i % 7)}_{i % 7}") for i in range(70)
    ]
    split = subject_independent_split(samples, ratios=(0.7, 0.0, 0.3), seed=0)
    assert_subject_independent(split)
    assert split.train and split.test


def test_leaked_interval_split_is_caught() -> None:
    leaked = _sample(0)
    with pytest.raises(SubjectLeakageError):
        assert_subject_independent(Split(train=[leaked], val=[], test=[leaked]))


def test_write_interval_split_persists_all_buckets_and_summary(tmp_path: Path) -> None:
    samples = [_sample(i, subject=f"gA_{i}") for i in range(10)]
    split = subject_independent_split(samples, ratios=(0.7, 0.0, 0.3), seed=0)
    write_interval_split(split, tmp_path, seed=0)
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "test.csv").exists()
    assert (tmp_path / "summary.json").exists()
    reread = read_interval_manifest(tmp_path / "train.csv")
    assert all(isinstance(s, IntervalSample) for s in reread)
    assert {s.subject_id for s in reread}.isdisjoint(
        {s.subject_id for s in read_interval_manifest(tmp_path / "test.csv")}
    )
