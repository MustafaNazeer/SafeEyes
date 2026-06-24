"""Seam tests for the MediaPipe FaceMesh wrapper.

The numeric geometry is tested elsewhere; here we cover the two seams that the
wrapper owns: the normalized-to-pixel mapping the geometry consumes, and the
constructor's guard for a missing model bundle.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from safeeyes.perception.facemesh import FaceMeshDetector, _to_pixel_array


@dataclass
class _StubLandmark:
    x: float
    y: float


def test_to_pixel_array_scales_normalized_coordinates() -> None:
    landmarks = [_StubLandmark(0.5, 0.5), _StubLandmark(0.0, 1.0), _StubLandmark(1.0, 0.25)]
    pixels = _to_pixel_array(landmarks, width=100, height=200)
    assert pixels.shape == (3, 2)
    np.testing.assert_allclose(
        pixels, [[50.0, 100.0], [0.0, 200.0], [100.0, 50.0]]
    )


def test_to_pixel_array_returns_float_array() -> None:
    pixels = _to_pixel_array([_StubLandmark(0.5, 0.5)], width=10, height=10)
    assert pixels.dtype == np.dtype(float)


def test_missing_model_file_raises_with_download_hint(tmp_path) -> None:
    pytest.importorskip("mediapipe")
    missing = tmp_path / "face_landmarker.task"
    with pytest.raises(FileNotFoundError) as excinfo:
        FaceMeshDetector(model_asset_path=missing)
    message = str(excinfo.value)
    assert str(missing) in message
    assert "face_landmarker.task" in message  # the download hint is included
