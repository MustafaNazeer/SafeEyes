"""Turn DMD drowsiness recordings into 3-class training windows.

The temporal model has three UTA-RLDD classes (alert, low vigilance, drowsy). DMD
drowsiness carries only a frame-level binary drowsy signal (a sustained eye
closure proxy), so its windows map onto two of the three classes: a window is
drowsy when a set fraction of its frames carry the drowsy label, otherwise it is
treated as alert. DMD contributes no low vigilance windows. Adding the DMD alert
windows is what teaches a model trained mostly on UTA to stop over-firing on the
DMD distribution.

Unlike the UTA harness, which gives every window of a clip the clip's single
label, DMD labels vary within a recording, so this builds per-window labels
directly rather than propagating one label per sequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from safeeyes.data.dmd_drowsiness import (
    DEFAULT_FPS,
    DEFAULT_MIN_CLOSED_SECONDS,
    frame_drowsy_labels,
)
from safeeyes.data.splits import Sample, subject_independent_split
from safeeyes.temporal.cross_eval import windowed_binary_labels

ALERT_CLASS = 0
DROWSY_CLASS = 2


def load_dmd_recording(
    features_dir: str | Path,
    ann_dir: str | Path,
    stem: str,
    *,
    fps: float = DEFAULT_FPS,
    min_closed_seconds: float = DEFAULT_MIN_CLOSED_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one DMD recording's features and the per-row drowsy labels aligned to them.

    Mirrors the alignment the cross-dataset evaluation uses: the frame-level drowsy
    labels are indexed by the kept frame indices, so each feature row gets the
    drowsy flag of the raw frame it came from.
    """
    data = np.load(Path(features_dir) / f"{stem}.npz")
    feats = np.asarray(data["features"], dtype=float)
    frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
    n_raw = int(data["n_raw_frames"])
    row_drowsy = frame_drowsy_labels(
        Path(ann_dir) / f"{stem}.json", n_raw, fps=fps, min_closed_seconds=min_closed_seconds
    )
    return feats, row_drowsy[frame_indices]


def load_dmd_train_windows(
    features_dir: str | Path,
    ann_dir: str | Path,
    stems: Sequence[str],
    *,
    window_size: int,
    stride: int,
    positive_fraction: float,
    fps: float = DEFAULT_FPS,
    min_closed_seconds: float = DEFAULT_MIN_CLOSED_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate 3-class training windows over several DMD recordings."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_features = 0
    for stem in stems:
        feats, row_drowsy = load_dmd_recording(
            features_dir, ann_dir, stem, fps=fps, min_closed_seconds=min_closed_seconds
        )
        if feats.ndim == 2:
            n_features = feats.shape[1]
        if feats.shape[0] < window_size:
            continue
        x, y = dmd_window_dataset(feats, row_drowsy, window_size, stride, positive_fraction)
        xs.append(x)
        ys.append(y)
    if xs:
        return np.concatenate(xs), np.concatenate(ys)
    return np.empty((0, window_size, n_features), dtype=float), np.empty((0,), dtype=int)


def dmd_subject_key(stem: str) -> str:
    """Subject identity from a DMD recording stem, the group and number (e.g. gB_10).

    Recordings of the same subject share this prefix, so splitting on it keeps all
    of a subject's recordings in one bucket.
    """
    parts = stem.split("_")
    return "_".join(parts[:2])


def dmd_subject_split(
    stems: Sequence[str],
    ratios: tuple[float, float, float] = (0.7, 0.0, 0.3),
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Split DMD recording stems into (train, test) with no subject on both sides."""
    samples = [Sample(sample_id=s, subject_id=dmd_subject_key(s), label="dmd") for s in stems]
    split = subject_independent_split(samples, ratios=ratios, seed=seed)
    return [s.sample_id for s in split.train], [s.sample_id for s in split.test]


def dmd_window_dataset(
    features: np.ndarray,
    row_drowsy: np.ndarray,
    window_size: int,
    stride: int,
    positive_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Window one DMD recording into (samples, window, features) with 3-class labels.

    A window is labeled drowsy when at least ``positive_fraction`` of its frames
    carry the frame-level drowsy label, otherwise alert. The window slices are the
    raw features, matching what the model consumes at serving time.
    """
    feats = np.asarray(features, dtype=float)
    n_features = feats.shape[1] if feats.ndim == 2 else 0
    windows, drowsy_flags = windowed_binary_labels(
        feats, row_drowsy, window_size, stride, positive_fraction
    )
    if windows:
        x = np.stack([np.asarray(w, dtype=float) for w in windows])
    else:
        x = np.empty((0, window_size, n_features), dtype=float)
    y = np.where(np.asarray(drowsy_flags, dtype=bool), DROWSY_CLASS, ALERT_CLASS).astype(int)
    return x, y
