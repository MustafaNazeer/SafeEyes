from pathlib import Path

import numpy as np

from safeeyes.perception.extract_gaze import assign_interval_ids, subject_id


def test_subject_id_is_the_group_and_subject_directories():
    video = Path("data/dmd/gaze/dmd/gF/23/s6/gF_23_s6_2019-03-04T16;04;39+01;00_rgb_face.mp4")
    assert subject_id(video) == "gF-23"


def test_two_recordings_of_one_subject_share_a_subject_id():
    """gF-23 and gZ-33 each have two s6 recordings; both belong to one person."""
    first = Path("data/dmd/gaze/dmd/gF/23/s6/gF_23_s6_2019-03-04T16;04;39+01;00_rgb_face.mp4")
    second = Path("data/dmd/gaze/dmd/gF/23/s6/gF_23_s6_2019-03-05T10;05;07+01;00_rgb_face.mp4")
    assert subject_id(first) == subject_id(second) == "gF-23"
    assert first.stem != second.stem


def test_contiguous_frames_of_one_zone_share_an_interval_id():
    ids = assign_interval_ids([0, 1, 2, 10, 11], ["front"] * 3 + ["left"] * 2)
    assert ids.tolist() == [0, 0, 0, 1, 1]


def test_a_gap_starts_a_new_interval():
    """A dropped frame splits an interval, since the glance may have ended."""
    ids = assign_interval_ids([0, 1, 5, 6], ["front"] * 4)
    assert ids.tolist() == [0, 0, 1, 1]


def test_a_zone_change_starts_a_new_interval():
    ids = assign_interval_ids([0, 1, 2, 3], ["front", "front", "left", "left"])
    assert ids.tolist() == [0, 0, 1, 1]


def test_returning_to_a_zone_is_a_distinct_interval():
    ids = assign_interval_ids([0, 1, 2, 3], ["front", "left", "front", "front"])
    assert ids.tolist() == [0, 1, 2, 2]


def test_a_single_frame_interval_is_counted():
    ids = assign_interval_ids([0, 5, 6], ["front", "left", "left"])
    assert ids.tolist() == [0, 1, 1]


def test_empty_input_yields_empty_ids():
    assert assign_interval_ids([], []).tolist() == []


def test_ids_are_contiguous_from_zero():
    ids = assign_interval_ids([0, 1, 4, 8, 9], ["a", "a", "b", "b", "b"])
    assert ids.tolist() == [0, 0, 1, 2, 2]
    assert set(ids.tolist()) == set(range(int(ids.max()) + 1))


def test_dtype_is_integer():
    assert np.issubdtype(assign_interval_ids([0, 1], ["a", "a"]).dtype, np.integer)
