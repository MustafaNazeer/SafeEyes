"""Interval labeled samples and their manifest persistence.

Extends the whole clip Sample with a frame interval so temporally annotated
datasets reuse the existing subject independent split machinery unchanged.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from safeeyes.data.manifest import _bucket_summary
from safeeyes.data.splits import Sample, Split

_HEADER = ("sample_id", "subject_id", "label", "start_frame", "end_frame")


@dataclass(frozen=True)
class IntervalSample(Sample):
    start_frame: int
    end_frame: int


def write_interval_manifest(samples: Sequence[IntervalSample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for s in samples:
            writer.writerow((s.sample_id, s.subject_id, s.label, s.start_frame, s.end_frame))


def read_interval_manifest(path: str | Path) -> list[IntervalSample]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = tuple(next(reader))
        if header != _HEADER:
            raise ValueError(f"unexpected interval manifest header: {header}")
        return [
            IntervalSample(
                sample_id=row[0],
                subject_id=row[1],
                label=row[2],
                start_frame=int(row[3]),
                end_frame=int(row[4]),
            )
            for row in reader
        ]


def write_interval_split(split: Split, out_dir: str | Path, seed: int) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = {"train": split.train, "val": split.val, "test": split.test}
    for name, samples in buckets.items():
        write_interval_manifest(
            [s for s in samples if isinstance(s, IntervalSample)], out_dir / f"{name}.csv"
        )
    summary = {
        "counts": {name: _bucket_summary(samples) for name, samples in buckets.items()},
        "seed": seed,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
