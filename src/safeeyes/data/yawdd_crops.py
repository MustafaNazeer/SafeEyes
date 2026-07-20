"""Mouth crop extraction from YawDD Mirror video, aligned to its own features.

An earlier feature pass over these videos wrote per frame geometry as bare
arrays with no record of which video frame each row came from. Because a frame
with no detected face is skipped, the row index and the video frame index drift
apart by an amount that cannot be recovered after the fact. This pass fixes
that: it records ``frame_indices`` alongside ``features``, so every row names
the exact decoded frame it was computed from, and it saves mouth crops keyed by
feature row rather than by video frame.

The four arrays a clip produces are mutually consistent by construction:

- ``features`` is ``(n, 5)``, one row per frame with a detected face.
- ``frame_indices`` is ``(n,)``, the video frame index each row came from.
- ``crop_rows`` is ``(m,)``, indices into ``features``, not into the video.
- ``crops`` is ``(m, size, size, 3)`` uint8, in ``crop_rows`` order.

Crops are kept only around frames whose mouth aspect ratio clears ``gate``,
plus ``margin_steps`` rows on each side so the onset and the release of a yawn
are covered, not only its peak. The gate default sits deliberately below the
preregistered yawn threshold in ``safeeyes.temporal.yawn_validation``: it is a
permissive pre filter over which frames are worth keeping pixels for, never a
decision about which frames are yawns.

Crops are written under the gitignored features tree. No frame or derivative of
a licensed dataset belongs in version control.

    python -m safeeyes.data.yawdd_crops \
        --manifest splits/yawdd/mirror-train.csv splits/yawdd/mirror-test.csv \
        --video-root <yawdd root> --out-root features/yawdd-crops --frame-step 3
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from safeeyes.data.manifest import read_manifest
from safeeyes.perception.extract import (
    FEATURE_DIM,
    FeatureFn,
    LandmarkDetector,
    _default_to_features,
    iter_video_frames,
)
from safeeyes.perception.frame import FEATURE_COLUMNS
from safeeyes.perception.mouth_crop import crop_mouth
from safeeyes.temporal.yawn_validation import CROP_MARGIN_STEPS, MAR_YAWN_THRESHOLD

__all__ = ["extract_clip_crops", "extract_manifest_crops", "extract_video_crops"]

# Bound to FEATURE_COLUMNS, the published source of truth for the feature
# column order, rather than a bare literal, so this module and the yawn model
# that reads its archives cannot drift apart on which column mar lives in.
# This module stays an import leaf: it imports FEATURE_COLUMNS, but nothing
# else in the package may import from this module.
MAR_COLUMN = FEATURE_COLUMNS.index("mar")
DEFAULT_GATE = 0.45


def extract_clip_crops(
    frames: Iterable[np.ndarray],
    detector: LandmarkDetector,
    *,
    to_features: FeatureFn | None = None,
    frame_step: int = 3,
    gate: float = DEFAULT_GATE,
    margin_steps: int = CROP_MARGIN_STEPS,
    size: int = 96,
) -> dict[str, np.ndarray]:
    """Features, their video frame indices, and mouth crops near gated rows.

    The gate decision for a row depends on a mouth aspect ratio that is only
    known once that row's features exist, and crops are wanted on both sides of
    a gated row. Rather than predict the decision from a ring buffer, every kept
    frame is cropped in the same loop iteration that produces its feature row,
    so a crop and its row are produced together and cannot drift; the selection
    by gate and margin then happens over indices alone, once the sequence is
    complete. A crop is 27 kB against a 0.9 MB frame, so holding one per row for
    a single clip costs a few megabytes.

    A frame whose mouth landmarks collapse to a degenerate box yields no crop.
    Its feature row is still kept, it simply never appears in ``crop_rows``.
    """
    if frame_step < 1:
        raise ValueError(f"frame_step must be positive, got {frame_step}")
    if margin_steps < 0:
        raise ValueError(f"margin_steps must be non negative, got {margin_steps}")
    if to_features is None:
        to_features = _default_to_features

    rows: list[np.ndarray] = []
    frame_indices: list[int] = []
    row_crops: list[np.ndarray | None] = []

    for index, frame in enumerate(frames):
        if index % frame_step != 0:
            continue
        landmarks = detector.landmarks(frame)
        if landmarks is None:
            continue
        height, width = frame.shape[:2]
        rows.append(np.asarray(to_features(landmarks, width, height), dtype=float))
        frame_indices.append(index)
        try:
            row_crops.append(crop_mouth(frame, landmarks, size=size))
        except ValueError:
            row_crops.append(None)

    features = np.vstack(rows) if rows else np.empty((0, FEATURE_DIM), dtype=float)
    indices = np.asarray(frame_indices, dtype=int).reshape(-1)

    n_rows = features.shape[0]
    keep = np.zeros(n_rows, dtype=bool)
    for row in np.flatnonzero(features[:, MAR_COLUMN] >= gate):
        low = max(0, int(row) - margin_steps)
        high = min(n_rows, int(row) + margin_steps + 1)
        keep[low:high] = True
    for position, crop in enumerate(row_crops):
        if crop is None:
            keep[position] = False

    crop_rows = np.flatnonzero(keep).astype(int)
    selected = [row_crops[int(row)] for row in crop_rows]
    if selected:
        # selected holds no None here: any row still marked keep already had
        # its crop cleared by the loop above when row_crops held None for it.
        # np.stack is called plainly, with no filtering comprehension, so a
        # future reordering that let a None slip through raises here instead
        # of silently shortening the crop stack against crop_rows.
        crops = np.stack(cast(list[np.ndarray], selected)).astype(np.uint8)
    else:
        crops = np.empty((0, size, size, 3), dtype=np.uint8)

    return {
        "features": features,
        "frame_indices": indices,
        "crop_rows": crop_rows,
        "crops": crops,
    }


def extract_video_crops(
    path: str | Path,
    detector: LandmarkDetector,
    **kwargs: object,
) -> dict[str, np.ndarray]:
    return extract_clip_crops(iter_video_frames(path), detector, **kwargs)  # type: ignore[arg-type]


def extract_manifest_crops(
    manifest_paths: Sequence[str | Path],
    video_root: str | Path,
    out_root: str | Path,
    detector: LandmarkDetector,
    *,
    to_features: FeatureFn | None = None,
    frame_step: int = 3,
    gate: float = DEFAULT_GATE,
    margin_steps: int = CROP_MARGIN_STEPS,
    size: int = 96,
    skip_existing: bool = True,
    limit: int | None = None,
    progress: Callable[[str, int], None] | None = None,
) -> list[Path]:
    """Extract and save one compressed archive of crops per manifest sample.

    Resumable in the same way as offline feature extraction: with
    ``skip_existing`` an archive already on disk is left alone, so an
    interrupted run picks up where it stopped.
    """
    video_root = Path(video_root)
    out_root = Path(out_root)
    written: list[Path] = []
    for manifest_path in manifest_paths:
        for sample in read_manifest(manifest_path):
            if limit is not None and len(written) >= limit:
                return written
            out_path = (out_root / sample.sample_id).with_suffix(".npz")
            if skip_existing and out_path.exists():
                written.append(out_path)
                continue
            result = extract_clip_crops(
                iter_video_frames(video_root / sample.sample_id),
                detector,
                to_features=to_features,
                frame_step=frame_step,
                gate=gate,
                margin_steps=margin_steps,
                size=size,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_path,
                features=result["features"],
                frame_indices=result["frame_indices"],
                crop_rows=result["crop_rows"],
                crops=result["crops"],
            )
            written.append(out_path)
            if progress is not None:
                progress(sample.sample_id, int(result["crop_rows"].shape[0]))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract mouth crops and aligned frame indices from video clips."
    )
    parser.add_argument("--manifest", required=True, nargs="+", help="one or more split manifests")
    parser.add_argument(
        "--video-root", required=True, help="root the manifest sample ids are under"
    )
    parser.add_argument("--out-root", required=True, help="directory to write crop archives")
    parser.add_argument("--frame-step", type=int, default=3, help="process every Nth frame")
    parser.add_argument(
        "--gate",
        type=float,
        default=DEFAULT_GATE,
        help=(
            "keep crops around frames whose mouth aspect ratio clears this value; "
            f"a permissive pre filter, deliberately below the preregistered yawn "
            f"threshold of {MAR_YAWN_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--margin-steps",
        type=int,
        default=CROP_MARGIN_STEPS,
        help="rows of crops kept on each side of a gate hit",
    )
    parser.add_argument("--size", type=int, default=96, help="square crop edge in pixels")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many samples, for throughput probes",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true", help="re-extract samples that already have crops"
    )
    args = parser.parse_args(argv)

    from safeeyes.perception.facemesh import FaceMeshDetector

    detector = FaceMeshDetector()
    try:
        written = extract_manifest_crops(
            args.manifest,
            args.video_root,
            args.out_root,
            detector,
            frame_step=args.frame_step,
            gate=args.gate,
            margin_steps=args.margin_steps,
            size=args.size,
            skip_existing=not args.no_skip_existing,
            limit=args.limit,
            progress=lambda sample_id, n: print(f"{sample_id}: {n} crops"),
        )
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()
    print(f"wrote {len(written)} crop archives under {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
