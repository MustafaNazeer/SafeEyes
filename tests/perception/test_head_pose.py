import cv2
import numpy as np
import pytest

from safeeyes.perception.head_pose import (
    CANONICAL_FACE_MODEL,
    default_camera_matrix,
    estimate_head_pose,
)


def _project(rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    image_points, _ = cv2.projectPoints(
        CANONICAL_FACE_MODEL, rvec, tvec, camera_matrix, np.zeros((4, 1))
    )
    return image_points.reshape(-1, 2)


def test_default_camera_matrix_shape_and_center() -> None:
    cm = default_camera_matrix(640, 480)
    assert cm.shape == (3, 3)
    assert cm[0, 2] == pytest.approx(320.0)
    assert cm[1, 2] == pytest.approx(240.0)


def test_frontal_face_recovers_near_zero_pose() -> None:
    cm = default_camera_matrix(640, 480)
    rvec = np.zeros((3, 1))
    tvec = np.array([[0.0], [0.0], [1000.0]])
    image_points = _project(rvec, tvec, cm)
    pitch, yaw, roll = estimate_head_pose(image_points, cm)
    assert abs(pitch) < 1.0
    assert abs(yaw) < 1.0
    assert abs(roll) < 1.0


def test_yaw_rotation_is_isolated_to_yaw() -> None:
    cm = default_camera_matrix(640, 480)
    # 20 degree rotation about the vertical axis.
    rvec = np.array([[0.0], [np.radians(20.0)], [0.0]])
    tvec = np.array([[0.0], [0.0], [1000.0]])
    image_points = _project(rvec, tvec, cm)
    pitch, yaw, roll = estimate_head_pose(image_points, cm)
    assert abs(yaw) == pytest.approx(20.0, abs=2.0)
    assert abs(pitch) < 2.0
    assert abs(roll) < 2.0


def test_wrong_point_count_raises() -> None:
    cm = default_camera_matrix(640, 480)
    with pytest.raises(ValueError):
        estimate_head_pose(np.zeros((5, 2)), cm)
