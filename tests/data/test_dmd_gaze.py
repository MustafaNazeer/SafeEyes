import pytest

from safeeyes.data.dmd_gaze import GAZE_ZONES, frame_gaze_labels, is_eyes_on_road


def _annotation(actions):
    return {
        "openlabel": {
            "actions": {
                str(index): {"type": action_type, "frame_intervals": intervals}
                for index, (action_type, intervals) in enumerate(actions)
            }
        }
    }


def test_zone_list_is_pinned_to_the_nine_valid_zones():
    assert GAZE_ZONES == (
        "left_mirror",
        "left",
        "front",
        "center_mirror",
        "front_right",
        "right_mirror",
        "right",
        "infotainment",
        "steering_wheel",
    )


def test_labels_expand_intervals_to_frames():
    ann = _annotation([("gaze_zone/front", [{"frame_start": 2, "frame_end": 4}])])
    assert frame_gaze_labels(ann, 10) == {2: "front", 3: "front", 4: "front"}


def test_multiple_intervals_of_one_zone_are_all_expanded():
    ann = _annotation(
        [
            (
                "gaze_zone/front",
                [{"frame_start": 0, "frame_end": 1}, {"frame_start": 5, "frame_end": 6}],
            )
        ]
    )
    assert frame_gaze_labels(ann, 10) == {0: "front", 1: "front", 5: "front", 6: "front"}


def test_not_valid_frames_are_excluded():
    ann = _annotation(
        [
            ("gaze_zone/front", [{"frame_start": 0, "frame_end": 1}]),
            ("gaze_zone/not_valid", [{"frame_start": 2, "frame_end": 5}]),
        ]
    )
    assert frame_gaze_labels(ann, 10) == {0: "front", 1: "front"}


def test_non_gaze_levels_are_ignored():
    ann = _annotation([("blinks/blinking", [{"frame_start": 0, "frame_end": 3}])])
    assert frame_gaze_labels(ann, 10) == {}


def test_intervals_are_clamped_to_the_video_length():
    """DMD annotations overrun video length; Phase 8 measured this in every session."""
    ann = _annotation([("gaze_zone/left", [{"frame_start": 8, "frame_end": 20}])])
    assert frame_gaze_labels(ann, 10) == {8: "left", 9: "left"}


def test_an_interval_starting_past_the_end_yields_nothing():
    ann = _annotation([("gaze_zone/left", [{"frame_start": 12, "frame_end": 20}])])
    assert frame_gaze_labels(ann, 10) == {}


def test_an_unknown_zone_raises_rather_than_being_silently_dropped():
    ann = _annotation([("gaze_zone/sunroof", [{"frame_start": 0, "frame_end": 1}])])
    with pytest.raises(ValueError, match="sunroof"):
        frame_gaze_labels(ann, 10)


def test_a_bare_root_without_the_openlabel_key_is_accepted():
    ann = _annotation([("gaze_zone/front", [{"frame_start": 0, "frame_end": 0}])])
    assert frame_gaze_labels(ann["openlabel"], 10) == {0: "front"}


def test_only_front_counts_as_eyes_on_road():
    assert is_eyes_on_road("front") is True
    for zone in GAZE_ZONES:
        if zone != "front":
            assert is_eyes_on_road(zone) is False
