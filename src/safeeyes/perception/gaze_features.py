"""The feature vector consumed by the gaze zone classifier.

Head pose supplies where the head is aimed; the iris offsets supply where the
eyes are aimed within it. Neither alone is sufficient: a driver can check a
mirror by turning the head, by moving the eyes, or by any mixture of the two.

Both come from landmarks the detector already produces for the drowsiness
path, so adding this signal costs no extra inference on the device.

The column order is fixed and published as GAZE_FEATURE_COLUMNS because the
exported model consumes the columns positionally, exactly as FEATURE_COLUMNS
does for the temporal path. A degenerate frame yields a neutral pose rather
than raising, so the live loop never crashes on a bad frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from safeeyes.perception.head_pose import estimate_head_pose
from safeeyes.perception.iris import iris_offset
from safeeyes.perception.landmarks import HEAD_POSE_INDICES, extract_points

GAZE_FEATURE_COLUMNS: tuple[str, ...] = (
    "pitch",
    "yaw",
    "roll",
    "left_iris_dx",
    "left_iris_dy",
    "right_iris_dx",
    "right_iris_dy",
)


def gaze_features(landmarks: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    try:
        head_points = extract_points(landmarks, HEAD_POSE_INDICES)
        pitch, yaw, roll = estimate_head_pose(head_points, camera_matrix)
    except (cv2.error, RuntimeError, ValueError):
        pitch, yaw, roll = 0.0, 0.0, 0.0

    left_dx, left_dy = iris_offset(landmarks, "left")
    right_dx, right_dy = iris_offset(landmarks, "right")

    return np.array(
        [pitch, yaw, roll, left_dx, left_dy, right_dx, right_dy],
        dtype=float,
    )
