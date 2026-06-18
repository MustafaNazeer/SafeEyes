"""Offline per frame feature extraction from video clips.

Drives the same perception path the live loop uses (FaceMesh landmarks then the
fixed five element feature vector) over recorded video instead of a camera, and
saves one feature array per clip for the temporal training harness to consume.
Frames where no face is found are skipped, exactly as the live loop skips them,
so the offline training features match what the runtime pipeline ever sees.

The output for a clip is an ``(n_frames, 5)`` array saved to
``feature_root/<sample_id>.npy``, the path layout
``safeeyes.temporal.train_temporal`` reads back.

    python -m safeeyes.perception.extract \
        --manifest splits/uta-rldd/train.csv splits/uta-rldd/test.csv \
        --video-root data/uta-rldd --feature-root features/uta-rldd
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from safeeyes.data.manifest import read_manifest
from safeeyes.perception.frame import FEATURE_COLUMNS, frame_features
from safeeyes.perception.head_pose import default_camera_matrix

FEATURE_DIM = len(FEATURE_COLUMNS)

FeatureFn = Callable[[np.ndarray, int, int], np.ndarray]


class LandmarkDetector(Protocol):
    def landmarks(self, frame: np.ndarray) -> np.ndarray | None: ...


def _default_to_features(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    return frame_features(landmarks, default_camera_matrix(width, height))


def extract_clip_features(
    frames: Iterable[np.ndarray],
    detector: LandmarkDetector,
    to_features: FeatureFn | None = None,
    frame_step: int = 1,
) -> np.ndarray:
    """Per frame features for one clip, shape (n_kept, FEATURE_DIM).

    Frames are subsampled by ``frame_step`` and any frame with no detected face
    is skipped, so the row count is the number of usable frames, not the raw
    frame count.
    """
    if to_features is None:
        to_features = _default_to_features
    rows: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        if frame_step > 1 and index % frame_step != 0:
            continue
        landmarks = detector.landmarks(frame)
        if landmarks is None:
            continue
        height, width = frame.shape[:2]
        rows.append(np.asarray(to_features(landmarks, width, height), dtype=float))
    if not rows:
        return np.empty((0, FEATURE_DIM), dtype=float)
    return np.vstack(rows)


def iter_video_frames(path: str | Path) -> Iterable[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def extract_video_features(
    path: str | Path,
    detector: LandmarkDetector,
    to_features: FeatureFn | None = None,
    frame_step: int = 1,
) -> np.ndarray:
    return extract_clip_features(iter_video_frames(path), detector, to_features, frame_step)


def extract_dataset_features(
    manifest_paths: Sequence[str | Path],
    video_root: str | Path,
    feature_root: str | Path,
    detector: LandmarkDetector,
    frame_step: int = 1,
    skip_existing: bool = True,
    progress: Callable[[str, int], None] | None = None,
) -> list[Path]:
    """Extract and save features for every clip across the given manifests.

    Resumable: with ``skip_existing`` an already written feature file is left in
    place, so an interrupted run picks up where it stopped.
    """
    video_root = Path(video_root)
    feature_root = Path(feature_root)
    written: list[Path] = []
    for manifest_path in manifest_paths:
        for sample in read_manifest(manifest_path):
            out_path = (feature_root / sample.sample_id).with_suffix(".npy")
            if skip_existing and out_path.exists():
                written.append(out_path)
                continue
            features = extract_video_features(
                video_root / sample.sample_id, detector, frame_step=frame_step
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, features)
            written.append(out_path)
            if progress is not None:
                progress(sample.sample_id, int(features.shape[0]))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract per frame features from video clips.")
    parser.add_argument("--manifest", required=True, nargs="+", help="one or more split manifests")
    parser.add_argument(
        "--video-root", required=True, help="root the manifest sample ids are under"
    )
    parser.add_argument("--feature-root", required=True, help="directory to write feature arrays")
    parser.add_argument(
        "--frame-step", type=int, default=1, help="process every Nth frame (default: every frame)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="re-extract clips that already have features",
    )
    args = parser.parse_args(argv)

    from safeeyes.perception.facemesh import FaceMeshDetector

    detector = FaceMeshDetector()
    try:
        written = extract_dataset_features(
            args.manifest,
            args.video_root,
            args.feature_root,
            detector,
            frame_step=args.frame_step,
            skip_existing=not args.no_skip_existing,
            progress=lambda sample_id, n: print(f"{sample_id}: {n} frames"),
        )
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()
    print(f"wrote {len(written)} feature arrays under {args.feature_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
