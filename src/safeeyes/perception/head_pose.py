"""Head pose estimation from facial landmarks.

A generic 3D face model is fitted to the detected 2D landmarks with solvePnP to
recover the head rotation, which is then expressed as pitch, yaw, and roll for
nod and head drop detection. The rotation to euler conversion lives in the
geometry module and is tested there; this module is the solvePnP wrapper and the
canonical model it fits against.
"""

from __future__ import annotations

import cv2
import numpy as np

from safeeyes.perception.geometry import rotation_matrix_to_euler

# Generic 3D face model in millimetres, ordered to match HEAD_POSE_INDICES:
# nose tip, chin, left eye outer corner, right eye outer corner, left and right
# mouth corners.
CANONICAL_FACE_MODEL: np.ndarray = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


def default_camera_matrix(width: int, height: int) -> np.ndarray:
    """A reasonable pinhole camera matrix when the camera is not calibrated."""
    focal = float(width)
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_head_pose(
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    model_points: np.ndarray = CANONICAL_FACE_MODEL,
) -> tuple[float, float, float]:
    points = np.ascontiguousarray(image_points, dtype=np.float64)
    if points.shape != (model_points.shape[0], 2):
        raise ValueError(
            f"expected image points of shape ({model_points.shape[0]}, 2), got {points.shape}"
        )
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rotation_vector, _ = cv2.solvePnP(
        model_points, points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("solvePnP failed to recover a head pose")

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return rotation_matrix_to_euler(rotation_matrix)
