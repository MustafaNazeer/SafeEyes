"""Build subject independent split manifests from a downloaded dataset.

This is the entry point a user runs once a dataset is on disk. It walks the
dataset into a manifest, partitions it so no subject crosses train and test,
verifies that property, and writes the fixed manifests and summary that every
reported metric traces back to. The heavy data stays where it is; only the
small manifests are produced.

    python -m safeeyes.data.build_splits \
        --dataset uta-rldd --root data/uta-rldd --out splits/uta-rldd
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from safeeyes.data.dmd_distraction import build_dmd_distraction_manifest
from safeeyes.data.intervals import write_interval_split
from safeeyes.data.manifest import write_split
from safeeyes.data.mrl import build_mrl_manifest
from safeeyes.data.splits import (
    Sample,
    Split,
    assert_subject_independent,
    subject_independent_split,
)
from safeeyes.data.uta_rldd import build_uta_manifest

DATASET_BUILDERS: dict[str, Callable[[Path], list[Sample]]] = {
    "uta-rldd": build_uta_manifest,
    "mrl": build_mrl_manifest,
    "dmd-distraction": cast(Callable[[Path], list[Sample]], build_dmd_distraction_manifest),
}

SPLIT_WRITERS: dict[str, Callable[[Split, Path, int], None]] = {
    "dmd-distraction": write_interval_split,
}


def build_dataset_splits(
    dataset: str,
    root: str | Path,
    out_dir: str | Path,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> Split:
    try:
        builder = DATASET_BUILDERS[dataset]
    except KeyError:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {sorted(DATASET_BUILDERS)}"
        ) from None
    manifest = builder(Path(root))
    if not manifest:
        raise ValueError(f"no samples found under {root!r}; check the dataset path and layout")
    split = subject_independent_split(manifest, ratios=ratios, seed=seed)
    assert_subject_independent(split)
    writer = SPLIT_WRITERS.get(dataset, write_split)
    writer(split, Path(out_dir), seed)
    return split


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build subject independent dataset splits.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_BUILDERS))
    parser.add_argument("--root", required=True, help="path to the downloaded dataset")
    parser.add_argument("--out", required=True, help="directory to write the split manifests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "TEST"),
        help="subject level train/val/test ratios (default: 0.7 0.15 0.15)",
    )
    args = parser.parse_args(argv)

    split = build_dataset_splits(
        dataset=args.dataset,
        root=args.root,
        out_dir=args.out,
        ratios=tuple(args.ratios),
        seed=args.seed,
    )
    print(
        f"wrote split to {args.out}: "
        f"{len(split.train)} train, {len(split.val)} val, {len(split.test)} test samples"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
