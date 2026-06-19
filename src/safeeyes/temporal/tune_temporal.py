"""Hyperparameter sweep for the temporal GRU on a held out validation fold.

Honest tuning. The validation fold is carved from the training subjects only, so
the test split is never touched during selection. Each config trains on the
reduced training subjects and is scored on the validation subjects; the best by
macro AUROC is reported. The chosen config is then trained on the full training
split and evaluated once on the test split, as a separate step, for the number
that gets reported.

    python -m safeeyes.temporal.tune_temporal \
        --train-manifest splits/uta-rldd/train.csv --feature-root features/uta-rldd
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from typing import cast

from safeeyes.data.manifest import read_manifest
from safeeyes.data.splits import assert_subject_independent, subject_independent_split
from safeeyes.temporal.train_temporal import (
    LabeledSequence,
    items_from_samples,
    train_and_evaluate,
)

EvaluateFn = Callable[..., dict[str, object]]

SWEEP_GRID: list[dict[str, object]] = [
    {"window_size": 100, "stride": 50, "epochs": 50, "lr": 1e-3},
    {"window_size": 150, "stride": 75, "epochs": 50, "lr": 1e-3},
    {"window_size": 200, "stride": 100, "epochs": 50, "lr": 1e-3},
    {"window_size": 150, "stride": 75, "epochs": 100, "lr": 1e-3},
    {"window_size": 150, "stride": 75, "epochs": 100, "lr": 5e-4},
    {"window_size": 100, "stride": 50, "epochs": 100, "lr": 1e-3},
    {"window_size": 200, "stride": 100, "epochs": 100, "lr": 1e-3},
    {"window_size": 150, "stride": 50, "epochs": 100, "lr": 1e-3},
]


def _default_evaluate(
    train: Sequence[LabeledSequence], val: Sequence[LabeledSequence], **cfg: object
) -> dict[str, object]:
    return train_and_evaluate(train, val, n_classes=3, **cfg)  # type: ignore[arg-type]


def run_sweep(
    train_items: Sequence[LabeledSequence],
    val_items: Sequence[LabeledSequence],
    grid: Sequence[dict[str, object]],
    evaluate: EvaluateFn = _default_evaluate,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    results: list[tuple[dict[str, object], dict[str, object]]] = []
    for cfg in grid:
        results.append((cfg, evaluate(train_items, val_items, **cfg)))
    return results


def best_by(
    results: Sequence[tuple[dict[str, object], dict[str, object]]], metric: str
) -> tuple[dict[str, object], dict[str, object]]:
    return max(results, key=lambda cr: cast(float, cr[1][metric]))


def _row(report: dict[str, object]) -> str:
    return (
        f"acc={cast(float, report['overall_accuracy']):.3f} "
        f"auroc={cast(float, report['macro_auroc']):.3f} "
        f"far={cast(float, report['false_alarm_rate']):.3f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune the temporal GRU on a validation fold.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument(
        "--seed", type=int, default=1, help="seed for the train and validation subject split"
    )
    args = parser.parse_args(argv)

    split = subject_independent_split(
        read_manifest(args.train_manifest),
        ratios=(1.0 - args.val_ratio, args.val_ratio, 0.0),
        seed=args.seed,
    )
    assert_subject_independent(split)
    train_items = items_from_samples(split.train, args.feature_root)
    val_items = items_from_samples(split.val, args.feature_root)
    print(f"tuning: {len(split.train)} train clips, {len(split.val)} val clips")

    results = run_sweep(train_items, val_items, SWEEP_GRID)
    for cfg, report in results:
        print(f"{_row(report)}  {json.dumps(cfg)}")

    best_cfg, best_report = best_by(results, "macro_auroc")
    print("\nBEST by val macro AUROC:")
    print(json.dumps(best_cfg))
    print(_row(best_report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
