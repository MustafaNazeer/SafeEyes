import math

import numpy as np
import pytest

from safeeyes.perception.geometry import (
    eye_aspect_ratio,
    mouth_aspect_ratio,
    perclos,
    rotation_matrix_to_euler,
)


def test_eye_aspect_ratio_known_value() -> None:
    # p1..p6 in order: horizontal corners p1,p4; vertical pairs (p2,p6) and (p3,p5).
    points = [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)]
    # (|p2-p6| + |p3-p5|) / (2 * |p1-p4|) = (2 + 2) / (2 * 4) = 0.5
    assert eye_aspect_ratio(points) == pytest.approx(0.5)


def test_eye_aspect_ratio_closed_eye_near_zero() -> None:
    points = [(0, 0), (1, 0.01), (3, 0.01), (4, 0), (3, -0.01), (1, -0.01)]
    assert eye_aspect_ratio(points) < 0.02


def test_eye_aspect_ratio_accepts_numpy_array() -> None:
    points = np.array([(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)], dtype=float)
    assert eye_aspect_ratio(points) == pytest.approx(0.5)


def test_mouth_aspect_ratio_known_value() -> None:
    # corners left,right; vertical pairs (top1,bottom1) and (top2,bottom2).
    points = [(0, 0), (2, 1), (4, 1), (6, 0), (4, -1), (2, -1)]
    # (|top1-bottom1| + |top2-bottom2|) / (2 * |left-right|) = (2 + 2) / (2 * 6)
    assert mouth_aspect_ratio(points) == pytest.approx(4 / 12)


def test_eye_aspect_ratio_degenerate_zero_width_is_zero() -> None:
    # A failed or degenerate detection collapses the points; EAR must not divide
    # by zero, it reports 0.0 (treated downstream as a closed or invalid eye).
    points = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    assert eye_aspect_ratio(points) == 0.0


def test_perclos_fraction_below_threshold() -> None:
    ears = [0.30, 0.05, 0.04, 0.28, 0.02]
    # three of five at or below 0.1
    assert perclos(ears, threshold=0.1) == pytest.approx(0.6)


def test_perclos_empty_is_zero() -> None:
    assert perclos([], threshold=0.1) == 0.0


def test_rotation_matrix_to_euler_identity_is_zero() -> None:
    pitch, yaw, roll = rotation_matrix_to_euler(np.eye(3))
    assert pitch == pytest.approx(0.0)
    assert yaw == pytest.approx(0.0)
    assert roll == pytest.approx(0.0)


def test_rotation_matrix_to_euler_known_yaw() -> None:
    # 30 degree rotation about the vertical (y) axis is a pure yaw.
    a = math.radians(30)
    ry = np.array(
        [
            [math.cos(a), 0, math.sin(a)],
            [0, 1, 0],
            [-math.sin(a), 0, math.cos(a)],
        ]
    )
    pitch, yaw, roll = rotation_matrix_to_euler(ry)
    assert yaw == pytest.approx(30.0, abs=1e-6)
    assert pitch == pytest.approx(0.0, abs=1e-6)
    assert roll == pytest.approx(0.0, abs=1e-6)
