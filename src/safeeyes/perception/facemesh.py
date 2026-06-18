"""MediaPipe FaceMesh runner.

A thin wrapper that turns a camera frame into a pixel coordinate landmark array
the geometry functions consume. MediaPipe is imported lazily so the rest of the
perception code, and its tests, do not depend on it. The numeric behaviour that
matters for correctness lives in the geometry and landmark modules, which are
tested directly; this wrapper is the integration seam exercised on real frames.

Recent MediaPipe builds removed the legacy ``solutions`` API, so this uses the
Tasks ``FaceLandmarker``. It produces the same canonical 478 point face mesh, so
the landmark indices used downstream are unchanged. The Tasks API needs a model
bundle; point it at ``face_landmarker.task`` via the constructor or the
``SAFEEYES_FACE_MODEL`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = os.environ.get("SAFEEYES_FACE_MODEL", "models/face_landmarker.task")

_DOWNLOAD_HINT = (
    "curl -L -o models/face_landmarker.task https://storage.googleapis.com/"
    "mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


class FaceMeshDetector:
    def __init__(
        self,
        model_asset_path: str | Path = DEFAULT_MODEL_PATH,
        num_faces: int = 1,
    ) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        path = str(model_asset_path)
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"FaceLandmarker model not found at {path!r}. Download it with:\n    "
                + _DOWNLOAD_HINT
            )
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            num_faces=num_faces,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def landmarks(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Return per landmark pixel coordinates (N, 2), or None if no face is found."""
        import cv2
        import mediapipe as mp

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        return np.array([(lm.x * width, lm.y * height) for lm in face], dtype=float)

    def close(self) -> None:
        self._landmarker.close()
