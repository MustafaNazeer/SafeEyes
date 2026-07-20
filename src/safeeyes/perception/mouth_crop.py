"""Square mouth crop geometry from face landmarks.

The crop follows the mean of the six mouth aspect ratio landmarks rather than
a static face box, so it tracks the mouth as the driver moves. The bounding
box is squared before the margin is applied so a downstream image classifier
never sees a distorted mouth. Clamping to the frame can make the final box
non square at a frame edge; that is accepted, because a distorted edge crop
is better than a crop that silently slides off the mouth.
"""

from __future__ import annotations

import numpy as np

from safeeyes.perception.landmarks import MOUTH_MAR_INDICES, extract_points


def mouth_crop_box(
    landmarks: np.ndarray, width: int, height: int, margin: float = 0.30
) -> tuple[int, int, int, int]:
    points = extract_points(landmarks, MOUTH_MAR_INDICES)[:, :2]
    cx = float(points[:, 0].mean())
    cy = float(points[:, 1].mean())
    span = float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1])))
    half = span * (1.0 + 2.0 * margin) / 2.0
    x0 = int(round(max(0.0, cx - half)))
    y0 = int(round(max(0.0, cy - half)))
    x1 = int(round(min(float(width), cx + half)))
    y1 = int(round(min(float(height), cy + half)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("degenerate mouth crop box")
    return x0, y0, x1, y1


def crop_mouth(
    frame: np.ndarray, landmarks: np.ndarray, margin: float = 0.30, size: int = 96
) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    x0, y0, x1, y1 = mouth_crop_box(landmarks, width, height, margin)
    patch = frame[y0:y1, x0:x1]
    resized: np.ndarray = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)
