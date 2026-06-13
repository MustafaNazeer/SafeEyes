import numpy as np
import pytest

from safeeyes.perception.frame import FEATURE_COLUMNS, frame_features
from safeeyes.perception.head_pose import default_camera_matrix
from safeeyes.perception.landmarks import (
    HEAD_POSE_INDICES,
    LEFT_EYE_EAR_INDICES,
    MOUTH_MAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
    average_eye_aspect_ratio,
    mouth_aspect_ratio_from_landmarks,
)


def _landmarks() -> np.ndarray:
    lm = np.zeros((468, 2), dtype=float)
    open_eye = [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)]  # EAR 0.5
    for idx, pt in zip(LEFT_EYE_EAR_INDICES, open_eye, strict=True):
        lm[idx] = pt
    for idx, pt in zip(RIGHT_EYE_EAR_INDICES, open_eye, strict=True):
        lm[idx] = pt
    mouth = [(0, 0), (2, 1), (4, 1), (6, 0), (4, -1), (2, -1)]  # MAR 4/12
    for idx, pt in zip(MOUTH_MAR_INDICES, mouth, strict=True):
        lm[idx] = pt
    head = [(320, 240), (320, 360), (260, 200), (380, 200), (280, 300), (360, 300)]
    for idx, pt in zip(HEAD_POSE_INDICES, head, strict=True):
        lm[idx] = pt
    return lm


def test_feature_columns_are_documented() -> None:
    assert FEATURE_COLUMNS == ("ear", "mar", "pitch", "yaw", "roll")


def test_frame_features_shape_and_composition() -> None:
    lm = _landmarks()
    feats = frame_features(lm, default_camera_matrix(640, 480))
    assert feats.shape == (5,)
    # the ear and mar columns come from the tested perception functions
    assert feats[0] == pytest.approx(average_eye_aspect_ratio(lm))
    assert feats[1] == pytest.approx(mouth_aspect_ratio_from_landmarks(lm))


def test_frame_features_head_pose_are_finite() -> None:
    feats = frame_features(_landmarks(), default_camera_matrix(640, 480))
    assert np.isfinite(feats[2:]).all()


def test_frame_features_degenerate_landmarks_do_not_crash() -> None:
    feats = frame_features(np.zeros((468, 2)), default_camera_matrix(640, 480))
    assert feats.shape == (5,)
    assert np.isfinite(feats).all()
