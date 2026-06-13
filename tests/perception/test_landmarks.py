import numpy as np
import pytest

from safeeyes.perception.landmarks import (
    LEFT_EYE_EAR_INDICES,
    MOUTH_MAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
    average_eye_aspect_ratio,
    extract_points,
    eye_aspect_ratios,
)


def _blank_landmarks(n: int = 468) -> np.ndarray:
    return np.zeros((n, 2), dtype=float)


def test_index_constants_have_six_points_each() -> None:
    assert len(LEFT_EYE_EAR_INDICES) == 6
    assert len(RIGHT_EYE_EAR_INDICES) == 6
    assert len(MOUTH_MAR_INDICES) == 6


def test_extract_points_selects_rows_in_order() -> None:
    landmarks = _blank_landmarks(10)
    landmarks[3] = (1.0, 2.0)
    landmarks[7] = (5.0, 6.0)
    out = extract_points(landmarks, (7, 3))
    assert out.tolist() == [[5.0, 6.0], [1.0, 2.0]]


def test_eye_aspect_ratios_reads_the_correct_indices() -> None:
    landmarks = _blank_landmarks()
    open_eye = [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)]  # EAR 0.5
    for idx, pt in zip(LEFT_EYE_EAR_INDICES, open_eye, strict=True):
        landmarks[idx] = pt
    left, right = eye_aspect_ratios(landmarks)
    assert left == pytest.approx(0.5)


def test_average_eye_aspect_ratio_is_mean_of_both_eyes() -> None:
    landmarks = _blank_landmarks()
    open_eye = [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)]  # EAR 0.5
    half_open = [(0, 0), (1, 0.5), (3, 0.5), (4, 0), (3, -0.5), (1, -0.5)]  # EAR 0.25
    for idx, pt in zip(LEFT_EYE_EAR_INDICES, open_eye, strict=True):
        landmarks[idx] = pt
    for idx, pt in zip(RIGHT_EYE_EAR_INDICES, half_open, strict=True):
        landmarks[idx] = pt
    assert average_eye_aspect_ratio(landmarks) == pytest.approx(0.375)


def test_extract_points_raises_on_out_of_range_index() -> None:
    with pytest.raises(IndexError):
        extract_points(_blank_landmarks(10), (50,))
