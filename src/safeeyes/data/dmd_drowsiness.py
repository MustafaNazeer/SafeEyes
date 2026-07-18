"""Frame-level drowsy labels from the DMD drowsiness OpenLABEL annotations.

The DMD drowsiness bundle annotates eye state as a temporal action, with sustained
eye closures (eyes_state/close) marked separately from blinks. For the cross
dataset evaluation we derive a per frame binary drowsy label: a frame is drowsy
when it lies inside an eyes_state/close interval that lasts at least a threshold
duration, a microsleep proxy. Ordinary blinks and brief closures are not drowsy,
and yawning is excluded by design, so the label isolates the sustained eye closure
signal the v1 fatigue model is meant to catch.

The drowsiness in this dataset was acted, not genuine fatigue, which is a real
limitation of any number derived from it and is stated wherever such a number is
reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from safeeyes.data.openlabel import parse_openlabel_actions

CLOSED_ACTION = "eyes_state/close"
DEFAULT_FPS = 29.76
DEFAULT_MIN_CLOSED_SECONDS = 0.5


def frame_drowsy_labels(
    source: dict[str, Any] | str | Path,
    n_frames: int,
    *,
    fps: float = DEFAULT_FPS,
    min_closed_seconds: float = DEFAULT_MIN_CLOSED_SECONDS,
) -> np.ndarray:
    """Per frame boolean drowsy labels derived from sustained eye closure.

    A frame is drowsy if it falls inside an ``eyes_state/close`` interval lasting
    at least ``min_closed_seconds``. Intervals are clamped to ``[0, n_frames)``,
    since DMD annotations can overrun the decoded video length.
    """
    min_frames = max(1, round(min_closed_seconds * fps))
    labels = np.zeros(n_frames, dtype=bool)
    for interval in parse_openlabel_actions(source):
        if interval.action != CLOSED_ACTION:
            continue
        if interval.end_frame - interval.start_frame + 1 < min_frames:
            continue
        lo = max(0, interval.start_frame)
        hi = min(n_frames - 1, interval.end_frame)
        if lo <= hi:
            labels[lo : hi + 1] = True
    return labels
