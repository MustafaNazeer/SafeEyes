import numpy as np

from safeeyes.perception.gaze_features import GAZE_FEATURE_COLUMNS, gaze_features
from safeeyes.perception.head_pose import default_camera_matrix
from safeeyes.perception.landmarks import LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES


def _face(n=478):
    """An upright frontal face with both eye sockets and irises populated."""
    points = np.zeros((n, 2), dtype=float)
    points[1] = (320.0, 250.0)
    points[152] = (320.0, 330.0)
    points[33] = (290.0, 230.0)
    points[263] = (350.0, 230.0)
    points[61] = (300.0, 290.0)
    points[291] = (340.0, 290.0)
    points[133] = (310.0, 230.0)
    points[362] = (330.0, 230.0)
    points[LEFT_IRIS_INDICES[0]] = (300.0, 230.0)
    points[RIGHT_IRIS_INDICES[0]] = (340.0, 230.0)
    return points


def test_column_order_is_pinned():
    """Pinned because the exported model consumes columns positionally."""
    assert GAZE_FEATURE_COLUMNS == (
        "pitch",
        "yaw",
        "roll",
        "left_iris_dx",
        "left_iris_dy",
        "right_iris_dx",
        "right_iris_dy",
    )


def test_feature_vector_length_matches_the_columns():
    vector = gaze_features(_face(), default_camera_matrix(640, 480))
    assert vector.shape == (len(GAZE_FEATURE_COLUMNS),)


def test_features_are_finite():
    vector = gaze_features(_face(), default_camera_matrix(640, 480))
    assert np.isfinite(vector).all()


def test_a_degenerate_frame_returns_finite_values_rather_than_raising():
    """The live loop must never crash on a bad frame, matching frame_features.

    Degenerate landmarks do not necessarily make solvePnP fail; it can converge
    on a meaningless pose instead. The guarantee is therefore that the call
    returns finite numbers, not that it returns zeros. Frames with no detected
    face never reach here, because the detector returns None upstream.
    """
    vector = gaze_features(np.zeros((478, 2), dtype=float), default_camera_matrix(640, 480))
    assert vector.shape == (len(GAZE_FEATURE_COLUMNS),)
    assert np.isfinite(vector).all()


def test_iris_columns_are_zero_when_the_socket_is_degenerate():
    vector = gaze_features(np.zeros((478, 2), dtype=float), default_camera_matrix(640, 480))
    assert vector[3:].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_iris_columns_track_the_iris_offset():
    face = _face()
    baseline = gaze_features(face, default_camera_matrix(640, 480))
    face[LEFT_IRIS_INDICES[0]] = (308.0, 230.0)
    shifted = gaze_features(face, default_camera_matrix(640, 480))
    assert shifted[3] > baseline[3]
    assert shifted[5] == baseline[5]


def test_a_precomputed_pose_is_reused_rather_than_solved_again():
    """The live loop already solves head pose for the drowsiness features.

    Solving it a second time on the same landmarks costs far more per frame
    than the gaze model itself, so the caller can pass the pose it already has.
    """
    face = _face()
    matrix = default_camera_matrix(640, 480)
    solved = gaze_features(face, matrix)
    reused = gaze_features(face, matrix, pose=(solved[0], solved[1], solved[2]))
    assert np.allclose(solved, reused)


def test_the_supplied_pose_is_what_lands_in_the_output():
    face = _face()
    vector = gaze_features(face, default_camera_matrix(640, 480), pose=(1.0, 2.0, 3.0))
    assert vector[:3].tolist() == [1.0, 2.0, 3.0]


def test_the_iris_columns_are_unaffected_by_the_supplied_pose():
    face = _face()
    matrix = default_camera_matrix(640, 480)
    solved = gaze_features(face, matrix)
    reused = gaze_features(face, matrix, pose=(9.0, 9.0, 9.0))
    assert np.allclose(solved[3:], reused[3:])
