"""MediaPipe FaceMesh landmark indices and extraction.

The geometry functions are deliberately library agnostic; this module is where
the MediaPipe FaceMesh topology lives. The eye indices follow the widely used
six point eye aspect ratio convention. The point ordering for every index tuple
is [p1, p2, p3, p4, p5, p6]: horizontal corners p1 and p4, with vertical pairs
(p2, p6) and (p3, p5), matching what the geometry functions expect.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from safeeyes.perception.geometry import eye_aspect_ratio, mouth_aspect_ratio

# Six point eye contours for the eye aspect ratio.
LEFT_EYE_EAR_INDICES: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR_INDICES: tuple[int, ...] = (362, 385, 387, 263, 373, 380)

# Mouth contour for the mouth aspect ratio: corners 61 and 291, with outer upper
# lip points (37, 267) paired against outer lower lip points (84, 314).
MOUTH_MAR_INDICES: tuple[int, ...] = (61, 37, 267, 291, 314, 84)

# Points used for head pose estimation via solvePnP: nose tip, chin, left and
# right eye outer corners, left and right mouth corners.
HEAD_POSE_INDICES: tuple[int, ...] = (1, 152, 33, 263, 61, 291)


def extract_points(landmarks: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return np.asarray(landmarks)[list(indices)]


def eye_aspect_ratios(landmarks: np.ndarray) -> tuple[float, float]:
    left = eye_aspect_ratio(extract_points(landmarks, LEFT_EYE_EAR_INDICES))
    right = eye_aspect_ratio(extract_points(landmarks, RIGHT_EYE_EAR_INDICES))
    return left, right


def average_eye_aspect_ratio(landmarks: np.ndarray) -> float:
    left, right = eye_aspect_ratios(landmarks)
    return (left + right) / 2.0


def mouth_aspect_ratio_from_landmarks(landmarks: np.ndarray) -> float:
    return mouth_aspect_ratio(extract_points(landmarks, MOUTH_MAR_INDICES))
