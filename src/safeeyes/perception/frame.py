"""Per frame feature vector.

Composes the tested perception signals into the single feature vector the
temporal window accumulates and the temporal classifier consumes. The column
order is fixed and published as FEATURE_COLUMNS so feature extraction and the
training harness agree on what each column means. A degenerate frame (no usable
landmarks, or a head pose that cannot be solved) yields a neutral vector rather
than raising, so the runtime loop never crashes on a bad frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from safeeyes.perception.head_pose import estimate_head_pose
from safeeyes.perception.landmarks import (
    HEAD_POSE_INDICES,
    average_eye_aspect_ratio,
    extract_points,
    mouth_aspect_ratio_from_landmarks,
)

FEATURE_COLUMNS: tuple[str, ...] = ("ear", "mar", "pitch", "yaw", "roll")


def frame_features(landmarks: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    ear = average_eye_aspect_ratio(landmarks)
    mar = mouth_aspect_ratio_from_landmarks(landmarks)
    try:
        head_points = extract_points(landmarks, HEAD_POSE_INDICES)
        pitch, yaw, roll = estimate_head_pose(head_points, camera_matrix)
    except (cv2.error, RuntimeError, ValueError):
        pitch, yaw, roll = 0.0, 0.0, 0.0
    return np.array([ear, mar, pitch, yaw, roll], dtype=float)
