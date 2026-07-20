"""The single frozen test run for the yawn detector.

Three detectors are scored on the same held out YawDD Mirror test subjects,
by the same code, at parameters fixed before this module existed:

* ``baseline_mar``, the plain geometric rule, a video fires if its mouth
  aspect ratio crosses the preregistered threshold at least once.
* ``baseline_duration``, the same rule plus a minimum event duration tuned on
  train subjects only.
* ``cnn``, the frozen backbone classifier over mouth crops, a video fires if
  any proposed event scores at or above the checkpoint's own tau.

The published geometric numbers were measured over the full Mirror
population, which is not this test split, so the baselines are recomputed
here restricted to the test subjects. Comparing the classifier against a
figure drawn from a different population would not be a like for like result.

Every row goes through ``score_detector``, which is a thin wrapper over
``precision_recall`` and ``score_videos`` in the geometric baseline modules.
The classifier reaches that shared code by folding each video's aggregated
score into a one row array, so a score at or above tau reads as exactly one
detected run: the same "fires or does not" rule the geometric rows use. No
detector gets a scoring function of its own.

Nothing in this module selects a threshold, a duration, or a tau. All three
arrive fixed, and the deploy bar in ``deploy_decision`` is numeric so the
comparison cannot be reinterpreted after the numbers are known.

    python -m safeeyes.models.eval_yawn \\
        --test-manifest splits/yawdd/mirror-test.csv \\
        --feature-root features/yawdd/mirror \\
        --crop-root features/yawdd-crops \\
        --checkpoint models/yawn.pt \\
        --min-duration 14 \\
        --out docs/ml/yawn-model-metrics.json
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
from safeeyes.models.train_distraction import build_transform, default_size
from safeeyes.models.yawn_decision import video_scores
from safeeyes.models.yawn_events import proposal_events
from safeeyes.models.yawn_model import (
    BACKBONE_NAME,
    _frozen_backbone,
    build_event_features,
    load_yawn_checkpoint,
    score_events,
)
from safeeyes.temporal.yawn_baseline import Video, load_mirror_mar, precision_recall
from safeeyes.temporal.yawn_validation import MAR_YAWN_THRESHOLD, score_videos

# The preregistered deploy bar. Both conditions must hold. These are constants,
# not tunables: they were written down before the test subjects were scored,
# and a run that moves them is not the run that was preregistered.
PRECISION_BAR = 0.70
RECALL_FLOOR = 0.90
# A classifier may trade recall for precision, but only so far. More than this
# much recall given up against the duration baseline does not count as beating
# it, however much precision it buys.
RECALL_DROP_ALLOWANCE = 0.05
# Pure float hygiene on the two boundary comparisons, not a slack parameter.
# It exists so a value that is exactly at the bar is not rejected by binary
# representation error alone.
_TOLERANCE = 1e-9


def deploy_decision(cnn: tuple[float, float], baseline: tuple[float, float]) -> str:
    """Apply the preregistered deploy bar to a measured result.

    ``cnn`` and ``baseline`` are each (precision, recall). Returns "deploy",
    or the name of the first condition that blocked it. The absolute bar is
    checked before the head to head comparison, so a result that fails both
    is reported against the bar it missed first.
    """
    cnn_precision, cnn_recall = cnn
    baseline_precision, baseline_recall = baseline

    if cnn_precision + _TOLERANCE < PRECISION_BAR:
        return "blocked_precision"
    if cnn_recall + _TOLERANCE < RECALL_FLOOR:
        return "blocked_recall"
    if cnn_precision <= baseline_precision:
        return "blocked_baseline"
    if baseline_recall - cnn_recall > RECALL_DROP_ALLOWANCE + _TOLERANCE:
        return "blocked_baseline"
    return "deploy"


def score_detector(
    videos: Sequence[Video], threshold: float, min_duration: int
) -> dict[str, object]:
    """One detector's row, measured by the shared geometric scoring code."""
    precision, recall, tp, fp, fn = precision_recall(videos, threshold, min_duration)
    summary = score_videos(
        [(label, mar) for _sample_id, label, mar in videos], threshold, min_duration
    )
    return {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "talking_false_positive_rate": summary["talking_false_positive_rate"],
    }


def evaluate_yawn(
    test_videos: Sequence[Video],
    crop_root: str | Path,
    checkpoint: str | Path,
    baseline_min_duration: int,
) -> dict[str, object]:
    """Score all three detectors on the same held out videos.

    ``test_videos`` carries the mouth aspect ratio series the geometric rows
    need. The classifier row is built from the crop archives for exactly the
    same sample ids, and every one of them stays in the denominator: a video
    whose mouth aspect ratio never crosses the threshold proposes no event,
    scores 0.0, and counts as a video that did not fire, never as a video that
    was dropped.
    """
    head, metadata = load_yawn_checkpoint(checkpoint)
    tau = float(metadata["tau"])
    n_frames = int(metadata["n_frames"])

    backbone, _dim = _frozen_backbone(BACKBONE_NAME)
    transform = build_transform(train=False, size=default_size(BACKBONE_NAME), normalize=True)

    # subject_id is a placeholder here. build_event_features only forwards it
    # onto YawnEvent, and nothing in this evaluation reads it: precision and
    # recall come from the manifest label, and subject level aggregation is not
    # part of the video level bar.
    samples = [
        Sample(sample_id=sample_id, subject_id="", label=label)
        for sample_id, label, _mar in test_videos
    ]
    features, _labels, events = build_event_features(
        samples,
        crop_root,
        backbone,
        transform,
        n_frames=n_frames,
        events_builder=proposal_events,
    )
    scores = score_events(head, features) if len(events) else np.empty((0,), dtype=np.float32)
    per_video = video_scores(events, scores)

    cnn_videos: list[Video] = [
        (sample_id, label, np.array([per_video.get(sample_id, 0.0)]))
        for sample_id, label, _mar in test_videos
    ]

    baseline_mar = score_detector(test_videos, MAR_YAWN_THRESHOLD, min_duration=1)
    baseline_duration = score_detector(
        test_videos, MAR_YAWN_THRESHOLD, min_duration=baseline_min_duration
    )
    cnn = score_detector(cnn_videos, tau, min_duration=1)

    decision = deploy_decision(
        cnn=(cast(float, cnn["precision"]), cast(float, cnn["recall"])),
        baseline=(
            cast(float, baseline_duration["precision"]),
            cast(float, baseline_duration["recall"]),
        ),
    )

    return {
        "baseline_mar": baseline_mar,
        "baseline_duration": baseline_duration,
        "cnn": cnn,
        "n_videos": len(test_videos),
        # Video tuples carry no subject id, so the caller that read the
        # manifest fills this in. It is declared here so the key order of the
        # published metrics file does not depend on who sets it.
        "n_subjects": None,
        "threshold": MAR_YAWN_THRESHOLD,
        "min_duration": baseline_min_duration,
        "tau": tau,
        "deploy_decision": decision,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score the yawn classifier and both geometric baselines on the test split."
    )
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--crop-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--min-duration", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    samples = read_manifest(args.test_manifest)
    test_videos = load_mirror_mar(args.test_manifest, args.feature_root)

    metrics = evaluate_yawn(
        test_videos, args.crop_root, args.checkpoint, baseline_min_duration=args.min_duration
    )
    metrics["n_subjects"] = len({sample.subject_id for sample in samples})
    metrics["test_manifest"] = str(args.test_manifest)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")

    for row in ("baseline_mar", "baseline_duration", "cnn"):
        entry = cast("dict[str, object]", metrics[row])
        print(
            f"{row:18s} precision={cast(float, entry['precision']):.4f} "
            f"recall={cast(float, entry['recall']):.4f} "
            f"tp={entry['tp']} fp={entry['fp']} fn={entry['fn']} "
            f"talking_fp_rate={entry['talking_false_positive_rate']}"
        )
    print(f"videos={metrics['n_videos']} subjects={metrics['n_subjects']} tau={metrics['tau']}")
    print(f"deploy_decision={metrics['deploy_decision']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
