import numpy as np
import pytest

from safeeyes.perception.iris import iris_offset
from safeeyes.perception.landmarks import (
    LEFT_EYE_EAR_INDICES,
    LEFT_IRIS_INDICES,
    RIGHT_EYE_EAR_INDICES,
    RIGHT_IRIS_INDICES,
)


def _eye(inner_x, outer_x, centre_y, iris_x, iris_y, eye="left", n=478, columns=2):
    """Build a synthetic face carrying one eye socket and its iris.

    The live detector returns (478, 2), so that is the default shape here.
    """
    points = np.zeros((n, columns), dtype=float)
    ear, iris = (
        (LEFT_EYE_EAR_INDICES, LEFT_IRIS_INDICES)
        if eye == "left"
        else (RIGHT_EYE_EAR_INDICES, RIGHT_IRIS_INDICES)
    )
    points[ear[0]][:2] = (inner_x, centre_y)
    points[ear[3]][:2] = (outer_x, centre_y)
    points[iris[0]][:2] = (iris_x, iris_y)
    return points


def test_centred_iris_has_zero_horizontal_offset():
    dx, _ = iris_offset(_eye(100.0, 140.0, 50.0, 120.0, 50.0), "left")
    assert abs(dx) < 1e-9


def test_iris_toward_the_first_corner_is_negative():
    dx, _ = iris_offset(_eye(100.0, 140.0, 50.0, 110.0, 50.0), "left")
    assert dx < 0


def test_iris_toward_the_far_corner_is_positive():
    dx, _ = iris_offset(_eye(100.0, 140.0, 50.0, 130.0, 50.0), "left")
    assert dx > 0


def test_offset_is_scale_invariant():
    """A face twice the size at twice the distance must give the same offset."""
    small = iris_offset(_eye(100.0, 140.0, 50.0, 110.0, 50.0), "left")
    large = iris_offset(_eye(200.0, 280.0, 100.0, 220.0, 100.0), "left")
    assert abs(small[0] - large[0]) < 1e-9


def test_vertical_offset_is_reported_and_signed():
    _, down = iris_offset(_eye(100.0, 140.0, 50.0, 120.0, 54.0), "left")
    _, up = iris_offset(_eye(100.0, 140.0, 50.0, 120.0, 46.0), "left")
    assert down > 0
    assert up < 0


def test_each_eye_reads_its_own_socket():
    """A left offset must not be computed from the right socket, or vice versa."""
    points = _eye(100.0, 140.0, 50.0, 110.0, 50.0, eye="left")
    right = _eye(300.0, 340.0, 50.0, 330.0, 50.0, eye="right")
    points[RIGHT_EYE_EAR_INDICES[0]] = right[RIGHT_EYE_EAR_INDICES[0]]
    points[RIGHT_EYE_EAR_INDICES[3]] = right[RIGHT_EYE_EAR_INDICES[3]]
    points[RIGHT_IRIS_INDICES[0]] = right[RIGHT_IRIS_INDICES[0]]

    assert iris_offset(points, "left")[0] < 0
    assert iris_offset(points, "right")[0] > 0


def test_a_degenerate_socket_returns_zero_rather_than_dividing_by_zero():
    assert iris_offset(_eye(100.0, 100.0, 50.0, 100.0, 50.0), "left") == (0.0, 0.0)


def test_three_column_landmarks_are_accepted():
    dx, _ = iris_offset(_eye(100.0, 140.0, 50.0, 110.0, 50.0, columns=3), "left")
    assert dx < 0


def test_unknown_eye_is_rejected():
    with pytest.raises(ValueError, match="middle"):
        iris_offset(_eye(100.0, 140.0, 50.0, 120.0, 50.0), "middle")
