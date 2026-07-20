"""Corpus wide coverage of the mouth crop extraction gate in yawdd_crops.py.

``yawdd_crops.py`` keeps a mouth crop only for feature rows whose mouth aspect
ratio clears a permissive gate (plus a margin on each side), so the crop
archives are not a uniform sample of a video's feature rows. An early estimate
of how uneven that is came from a two clip probe during development and was
never checked against the rest of the corpus. This module measures it for
real: for every sample in a manifest, it reads the sample's crop archive,
counts its total feature rows and its retained crop rows, and sums both by the
manifest's label column, so the reported fraction is a true pooled rate over
every video in a category rather than an average of per video ratios.

    python -m safeeyes.data.crop_coverage \\
        --manifest splits/yawdd/mirror.csv \\
        --crop-root features/yawdd-crops \\
        --out docs/ml/yawdd-crop-coverage.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

from safeeyes.data.manifest import read_manifest
from safeeyes.data.splits import Sample

__all__ = ["coverage_table", "crop_coverage", "load_row_counts"]


def load_row_counts(sample_id: str, crop_root: str | Path) -> tuple[int, int]:
    """A sample's (feature_rows, crop_rows), read from its crop archive.

    The archive path mirrors exactly how ``yawdd_crops.extract_manifest_crops``
    names its output: the sample id, suffixed ``.npz``, under ``crop_root``.
    """
    path = (Path(crop_root) / sample_id).with_suffix(".npz")
    with np.load(path) as data:
        return int(data["features"].shape[0]), int(data["crop_rows"].shape[0])


def coverage_table(rows: Sequence[tuple[str, int, int]]) -> dict[str, object]:
    """Pool (label, feature_rows, crop_rows) rows into a per category table.

    Each category's retained fraction is its summed crop rows over its summed
    feature rows, a pooled rate across every sample in the category, not an
    average of individual per sample ratios. ``all`` pools across every
    category present in ``rows``, so it can never silently drop one.
    """
    counts: dict[str, dict[str, int]] = {}
    for label, feature_rows, crop_rows in rows:
        if feature_rows < 0 or crop_rows < 0:
            raise ValueError(
                f"row counts must be non negative, got feature_rows={feature_rows} "
                f"crop_rows={crop_rows} for label {label!r}"
            )
        if crop_rows > feature_rows:
            raise ValueError(
                f"crop_rows ({crop_rows}) cannot exceed feature_rows ({feature_rows}) "
                f"for label {label!r}"
            )
        entry = counts.setdefault(label, {"samples": 0, "feature_rows": 0, "crop_rows": 0})
        entry["samples"] += 1
        entry["feature_rows"] += feature_rows
        entry["crop_rows"] += crop_rows

    per_category: dict[str, dict[str, object]] = {}
    total_samples = 0
    total_feature_rows = 0
    total_crop_rows = 0
    for label, entry in counts.items():
        feature_total = entry["feature_rows"]
        crop_total = entry["crop_rows"]
        per_category[label] = {
            "samples": entry["samples"],
            "feature_rows": feature_total,
            "crop_rows": crop_total,
            "retained_fraction": (crop_total / feature_total) if feature_total else None,
        }
        total_samples += entry["samples"]
        total_feature_rows += feature_total
        total_crop_rows += crop_total

    return {
        "per_category": per_category,
        "all": {
            "samples": total_samples,
            "feature_rows": total_feature_rows,
            "crop_rows": total_crop_rows,
            "retained_fraction": (
                (total_crop_rows / total_feature_rows) if total_feature_rows else None
            ),
        },
    }


def crop_coverage(samples: Sequence[Sample], crop_root: str | Path) -> dict[str, object]:
    """The coverage table for every sample's crop archive, grouped by label."""
    rows = [(sample.label, *load_row_counts(sample.sample_id, crop_root)) for sample in samples]
    return coverage_table(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how many feature rows the mouth crop extraction gate retains, "
            "per manifest category, across the full crop archive corpus."
        )
    )
    parser.add_argument("--manifest", required=True, help="the manifest listing every sample")
    parser.add_argument("--crop-root", required=True, help="directory holding the crop archives")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    samples = read_manifest(args.manifest)
    result = crop_coverage(samples, args.crop_root)
    result["manifest"] = str(args.manifest)
    result["crop_root"] = str(args.crop_root)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    per_category = cast("dict[str, dict[str, object]]", result["per_category"])
    for label, entry in per_category.items():
        fraction = cast("float | None", entry["retained_fraction"])
        rendered = f"{fraction:.4f}" if fraction is not None else "n/a"
        print(
            f"{label:20s} crop_rows={entry['crop_rows']} feature_rows={entry['feature_rows']} "
            f"retained_fraction={rendered}"
        )
    all_entry = cast("dict[str, object]", result["all"])
    all_fraction = cast(float, all_entry["retained_fraction"])
    print(
        "all                  "
        f"crop_rows={all_entry['crop_rows']} feature_rows={all_entry['feature_rows']} "
        f"retained_fraction={all_fraction:.4f}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
