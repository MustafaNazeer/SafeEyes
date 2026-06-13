"""Reading and writing split manifests.

A split is only reproducible if it is written down. These manifests are the
fixed record that every reported metric traces back to: small CSV files of
(sample_id, subject_id, label) plus a JSON summary recording the seed and the
class distribution per bucket. The manifests are tracked; the heavy data they
point at is not.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from safeeyes.data.splits import Sample, Split, class_distribution

_HEADER = ("sample_id", "subject_id", "label")


def write_manifest(samples: Sequence[Sample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for s in samples:
            writer.writerow((s.sample_id, s.subject_id, s.label))


def read_manifest(path: str | Path) -> list[Sample]:
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            Sample(sample_id=row["sample_id"], subject_id=row["subject_id"], label=row["label"])
            for row in reader
        ]


def _bucket_summary(samples: Sequence[Sample]) -> dict[str, object]:
    return {
        "samples": len(samples),
        "subjects": len({s.subject_id for s in samples}),
        "class_distribution": class_distribution(samples),
    }


def write_split(split: Split, out_dir: str | Path, seed: int) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(split.train, out_dir / "train.csv")
    write_manifest(split.val, out_dir / "val.csv")
    write_manifest(split.test, out_dir / "test.csv")
    summary = {
        "seed": seed,
        "counts": {
            "train": _bucket_summary(split.train),
            "val": _bucket_summary(split.val),
            "test": _bucket_summary(split.test),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
