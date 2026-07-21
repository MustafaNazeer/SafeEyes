"""Fixed length rolling feature window.

Per frame feature vectors accumulate here. The window holds the most recent
`capacity` frames in arrival order; once full, the oldest frame is evicted as a
new one arrives. The temporal classifier consumes the window as a (frames,
features) array. The contract is intentionally small and fully tested so the
buffer cannot silently drop, reorder, or grow beyond its bound.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np


def make_windows(
    sequence: np.ndarray, label: int, window_size: int, stride: int
) -> tuple[list[np.ndarray], list[int]]:
    """Slice a per frame feature sequence into fixed length training windows.

    Only complete windows are kept; a trailing partial window is dropped. Each
    window inherits the sequence label, which is how a video level UTA-RLDD label
    is propagated to the windows cut from it.
    """
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    seq = np.asarray(sequence, dtype=float)
    windows: list[np.ndarray] = []
    start = 0
    while start + window_size <= seq.shape[0]:
        windows.append(seq[start : start + window_size])
        start += stride
    return windows, [label] * len(windows)


def assemble_windowed_dataset(
    items: Sequence[tuple[np.ndarray, int]], window_size: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Window many labeled sequences and stack them into (samples, window, features)."""
    all_windows: list[np.ndarray] = []
    all_labels: list[int] = []
    n_features = 0
    for sequence, label in items:
        seq = np.asarray(sequence, dtype=float)
        if seq.ndim == 2:
            n_features = seq.shape[1]
        windows, labels = make_windows(seq, label, window_size, stride)
        all_windows.extend(windows)
        all_labels.extend(labels)
    if all_windows:
        x = np.stack(all_windows)
    else:
        x = np.empty((0, window_size, n_features), dtype=float)
    return x, np.asarray(all_labels, dtype=int)


class FeatureWindow:
    def __init__(self, capacity: int, n_features: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self._capacity = capacity
        self._n_features = n_features
        self._frames: deque[list[float]] = deque(maxlen=capacity)

    def push(self, features: Sequence[float] | np.ndarray) -> None:
        if len(features) != self._n_features:
            raise ValueError(f"expected {self._n_features} features, got {len(features)}")
        self._frames.append([float(v) for v in features])

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def is_full(self) -> bool:
        return len(self._frames) == self._capacity

    def as_array(self) -> np.ndarray:
        if not self._frames:
            return np.empty((0, self._n_features), dtype=float)
        return np.array(self._frames, dtype=float)

    def clear(self) -> None:
        self._frames.clear()
