"""Iris position within the eye socket.

Head pose says where the head is aimed but not where the eyes are aimed within
it, so a glance made with the eyes alone is invisible to pose. That is exactly
the distinction the gaze zones turn on, since looking front and looking front
right can share a head position.

The offset is normalized by socket width, so it is invariant to face size and
camera distance: the same glance gives the same number whether the driver sits
near the camera or far from it.
"""

from __future__ import annotations

import numpy as np

from safeeyes.perception.landmarks import (
    LEFT_EYE_EAR_INDICES,
    LEFT_IRIS_INDICES,
    RIGHT_EYE_EAR_INDICES,
    RIGHT_IRIS_INDICES,
)

_EYES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "left": (LEFT_EYE_EAR_INDICES, LEFT_IRIS_INDICES),
    "right": (RIGHT_EYE_EAR_INDICES, RIGHT_IRIS_INDICES),
}


def iris_offset(landmarks: np.ndarray, eye: str) -> tuple[float, float]:
    """Iris centre offset from the socket centre, in units of socket width.

    Returns (dx, dy). Positive dx is toward the fourth eye contour point, and
    positive dy is downward, following image coordinates.
    """
    if eye not in _EYES:
        raise ValueError(f"eye must be 'left' or 'right', got {eye!r}")
    ear_indices, iris_indices = _EYES[eye]
    points = np.asarray(landmarks, dtype=float)

    first = points[ear_indices[0]][:2]
    fourth = points[ear_indices[3]][:2]
    iris = points[iris_indices[0]][:2]

    width = float(np.linalg.norm(fourth - first))
    if width <= 0.0:
        return 0.0, 0.0

    centre = (first + fourth) / 2.0
    delta = (iris - centre) / width
    return float(delta[0]), float(delta[1])
