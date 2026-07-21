"""Extract gaze features and labels from the DMD gaze recordings.

This is a dedicated pass rather than a reuse of ``extract_clip_features``,
because that function drops frames with no detected face without recording
which frames it kept. Gaze labels are per frame, so a row that cannot be traced
back to its frame index cannot be labelled at all. Every row here carries its
frame index for exactly that reason.

Every valid frame is kept. There is no cadence to match: the gaze classifier
scores one frame at a time and has no window, unlike the temporal model whose
window has to span the same wall clock duration in training and serving.

Frames inside one annotation interval are near duplicates, so each row also
carries an interval id. Evaluation reports an interval level number beside the
frame level one, since the interval, not the frame, is the independent unit.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from safeeyes.data.dmd_gaze import frame_gaze_labels
from safeeyes.perception.extract import iter_video_frames
from safeeyes.perception.facemesh import FaceMeshDetector
from safeeyes.perception.gaze_features import GAZE_FEATURE_COLUMNS, gaze_features
from safeeyes.perception.head_pose import default_camera_matrix

GAZE_FEATURE_DIM = len(GAZE_FEATURE_COLUMNS)


def assign_interval_ids(frames: Sequence[int], zones: Sequence[str]) -> np.ndarray:
    """Group rows into the annotation intervals they came from.

    A new interval starts whenever the zone changes or the frame index is not
    consecutive with the previous row. A gap means frames were dropped between
    them, so the two runs cannot be assumed to belong to one continuous glance.
    """
    ids: list[int] = []
    current = -1
    previous: tuple[int, str] | None = None
    for frame, zone in zip(frames, zones, strict=True):
        starts_new_interval = previous is None or zone != previous[1] or frame != previous[0] + 1
        if starts_new_interval:
            current += 1
        ids.append(current)
        previous = (frame, zone)
    return np.array(ids, dtype=int)


def subject_id(video_path: Path) -> str:
    """The split unit, derived from the group and subject directories.

    Two subjects have more than one s6 recording, so a recording is not the
    same thing as a subject. Splits are built on this value to keep the same
    person out of two folds.
    """
    return f"{video_path.parents[2].name}-{video_path.parents[1].name}"


def extract_recording(
    video_path: str | Path,
    annotation_path: str | Path,
    detector: FaceMeshDetector,
) -> dict[str, np.ndarray]:
    with open(annotation_path) as handle:
        annotation = json.load(handle)

    # The video length is not known until it is decoded, so labels are built
    # unclamped and the overrun is accounted for afterwards. Frames annotated
    # past the end simply never match a decoded index.
    labels = frame_gaze_labels(annotation, 10**9)

    rows: list[np.ndarray] = []
    kept_frames: list[int] = []
    kept_zones: list[str] = []
    total = 0

    for index, frame in enumerate(iter_video_frames(video_path)):
        total = index + 1
        zone = labels.get(index)
        if zone is None:
            continue
        landmarks = detector.landmarks(frame)
        if landmarks is None:
            continue
        height, width = frame.shape[:2]
        rows.append(gaze_features(landmarks, default_camera_matrix(width, height)))
        kept_frames.append(index)
        kept_zones.append(zone)

    if total == 0:
        raise ValueError(f"no frames decoded from {video_path}")

    in_range = sum(1 for index in labels if index < total)
    features = np.vstack(rows) if rows else np.empty((0, GAZE_FEATURE_DIM), dtype=float)
    return {
        "features": features,
        # Unicode rather than object dtype so the archives load without
        # allow_pickle, which would otherwise make reading a feature file a
        # code execution path.
        "labels": np.array(kept_zones, dtype="U"),
        "frame_indices": np.array(kept_frames, dtype=int),
        "interval_ids": assign_interval_ids(kept_frames, kept_zones),
        "n_frames_total": np.array(total, dtype=int),
        "n_frames_labelled": np.array(in_range, dtype=int),
        "n_labels_past_end": np.array(len(labels) - in_range, dtype=int),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract gaze features from DMD recordings.")
    parser.add_argument("--gaze-root", required=True, help="extracted DMD gaze tree")
    parser.add_argument("--feature-root", required=True, help="destination for the npz files")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    out_root = Path(args.feature_root)
    out_root.mkdir(parents=True, exist_ok=True)
    videos = sorted(Path(args.gaze_root).rglob("*_rgb_face.mp4"))
    if not videos:
        raise FileNotFoundError(f"no face videos under {args.gaze_root}")

    detector = FaceMeshDetector()
    try:
        for index, video in enumerate(videos, start=1):
            subject = subject_id(video)
            # Named by the recording, not the subject: gF-23 and gZ-33 each have
            # two s6 recordings, so keying on the subject would have the second
            # silently overwrite the first. The subject is stored inside the file
            # instead, because it is the split unit.
            out_path = out_root / f"{video.stem}.npz"
            if args.skip_existing and out_path.exists():
                print(f"[{index}/{len(videos)}] {video.stem} exists, skipping", flush=True)
                continue
            annotation = video.with_name(video.name.replace("_rgb_face.mp4", "_rgb_ann_gaze.json"))
            if not annotation.exists():
                raise FileNotFoundError(f"missing annotation for {video}")
            result = extract_recording(video, annotation, detector)
            result["subject"] = np.array(subject, dtype="U")
            np.savez_compressed(out_path, **result)
            print(
                f"[{index}/{len(videos)}] {subject}: {len(result['frame_indices'])} rows "
                f"from {int(result['n_frames_labelled'])} labelled frames",
                flush=True,
            )
    finally:
        detector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
