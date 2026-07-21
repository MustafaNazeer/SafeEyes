"""Subject independent train and test manifests for the YawDD Mirror set.

The Mirror set carries 90 subjects with 2 to 7 videos each, so a random video
level split would leak a subject across both sides and inflate every number.
Ratios are fixed at 0.78 and 0.22 so a 90 subject population lands on exactly
70 train and 20 test subjects. splits/yawdd is gitignored, so these manifests
regenerate rather than ship.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from safeeyes.data.manifest import read_manifest, write_manifest
from safeeyes.data.splits import (
    Sample,
    Split,
    assert_subject_independent,
    subject_independent_split,
)

MIRROR_RATIOS = (0.78, 0.0, 0.22)
VALIDATION_RATIOS = (0.8, 0.2, 0.0)


def build_mirror_split(
    manifest_path: str | Path,
    out_dir: str | Path,
    seed: int = 0,
    expected_train_subjects: int = 70,
    expected_test_subjects: int = 20,
) -> Split:
    samples = read_manifest(manifest_path)
    split = subject_independent_split(samples, ratios=MIRROR_RATIOS, seed=seed)
    assert_subject_independent(split)

    actual_train_subjects = len({s.subject_id for s in split.train})
    actual_test_subjects = len({s.subject_id for s in split.test})
    if (
        actual_train_subjects != expected_train_subjects
        or actual_test_subjects != expected_test_subjects
    ):
        raise ValueError(
            "mirror split subject counts do not match the expected contract: "
            f"expected {expected_train_subjects} train and {expected_test_subjects} test, "
            f"got {actual_train_subjects} train and {actual_test_subjects} test"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(split.train, out_dir / "mirror-train.csv")
    write_manifest(split.test, out_dir / "mirror-test.csv")
    return split


def carve_validation(train: Sequence[Sample], seed: int = 0) -> tuple[list[Sample], list[Sample]]:
    inner = subject_independent_split(train, ratios=VALIDATION_RATIOS, seed=seed)
    assert_subject_independent(inner)
    return inner.train, inner.val


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the subject independent YawDD Mirror split."
    )
    parser.add_argument("--manifest", required=True, help="path to the mirror manifest CSV")
    parser.add_argument("--out-dir", required=True, help="directory to write the split manifests")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    split = build_mirror_split(args.manifest, args.out_dir, seed=args.seed)
    print(
        f"wrote split to {args.out_dir}: {len(split.train)} train, {len(split.test)} test samples"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
