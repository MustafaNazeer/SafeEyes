"""MediaPipe FaceMesh runner.

A thin wrapper that turns a camera frame into a pixel coordinate landmark array
the geometry functions can consume. MediaPipe is imported lazily so the rest of
the perception code, and its tests, do not depend on it. The numeric behaviour
that matters for correctness lives in the geometry and landmark modules, which
are tested directly; this wrapper is the integration seam exercised on real
frames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


class FaceMeshDetector:
    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def landmarks(self, frame_bgr: npt.NDArray[np.uint8]) -> np.ndarray | None:
        """Return per landmark pixel coordinates (N, 2), or None if no face is found."""
        import cv2

        height, width = frame_bgr.shape[:2]
        result = self._mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return None
        face = result.multi_face_landmarks[0]
        return np.array(
            [(lm.x * width, lm.y * height) for lm in face.landmark],
            dtype=float,
        )

    def close(self) -> None:
        self._mesh.close()
