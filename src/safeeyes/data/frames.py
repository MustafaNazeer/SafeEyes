"""Frame sampling for video preprocessing.

The drowsiness videos run several minutes at full frame rate, far denser than
the temporal features need. Sampling is split into a pure index calculation,
which is the part with real logic and is fully tested, and a thin reader that
pulls those frames off disk. Keeping the calculation pure means the sampling
behaviour is verified without decoding a single real video.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import numpy as np

T = TypeVar("T")


def sample_frame_indices(total_frames: int, source_fps: float, target_fps: float) -> list[int]:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if target_fps >= source_fps:
        return list(range(total_frames))
    step = source_fps / target_fps
    count = int(total_frames / step)
    return [round(i * step) for i in range(count)]


def extract_frames(read_frame: Callable[[int], T], indices: Sequence[int]) -> list[T]:
    return [read_frame(i) for i in indices]


def extract_sampled_frames(video_path: str | Path, target_fps: float) -> list[np.ndarray]:
    """Decode a video and return frames sampled down to target_fps.

    Exercised on real videos during preprocessing. The sampling decision is
    delegated to the tested pure functions above.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(total_frames, source_fps, target_fps)

        def read_frame(index: int) -> np.ndarray:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"failed to read frame {index} from {video_path}")
            return frame  # type: ignore[no-any-return]

        return extract_frames(read_frame, indices)
    finally:
        capture.release()
