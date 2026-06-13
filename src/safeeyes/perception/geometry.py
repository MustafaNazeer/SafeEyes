"""Interpretable per frame geometry.

These are the correctness critical signals the whole detector rests on, so they
are kept as small pure functions over point coordinates with no dependency on
any landmark library. The aspect ratios follow the standard eye aspect ratio
formulation: the mean of the vertical openings divided by the horizontal width.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Point = tuple[float, float]
Points = Sequence[Point] | np.ndarray


def _aspect_ratio(points: Points) -> float:
    """Aspect ratio over six points ordered [p1, p2, p3, p4, p5, p6].

    p1 and p4 are the horizontal corners; (p2, p6) and (p3, p5) are the two
    vertical pairs. Returns (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|).
    """
    p = np.asarray(points, dtype=float)
    if p.shape != (6, 2):
        raise ValueError(f"expected 6 (x, y) points, got shape {p.shape}")
    vertical = float(np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4]))
    horizontal = float(np.linalg.norm(p[0] - p[3]))
    if horizontal < 1e-9:
        return 0.0
    return vertical / (2.0 * horizontal)


def eye_aspect_ratio(points: Points) -> float:
    return _aspect_ratio(points)


def mouth_aspect_ratio(points: Points) -> float:
    return _aspect_ratio(points)


def perclos(ear_values: Sequence[float] | np.ndarray, threshold: float) -> float:
    """Proportion of eye closure: the fraction of values at or below threshold."""
    arr = np.asarray(ear_values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr <= threshold))


def rotation_matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """Decompose a rotation matrix into (pitch, yaw, roll) in degrees."""
    r = np.asarray(rotation, dtype=float)
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy < 1e-6:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)
