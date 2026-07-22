import numpy as np

from safeeyes.alert.hud import draw_hud
from safeeyes.alert.state_machine import AlertTier


def test_draw_hud_returns_same_shape_for_every_tier() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for tier in AlertTier:
        out = draw_hud(frame, tier)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8


def test_draw_hud_does_not_mutate_input() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    out = draw_hud(frame, AlertTier.ALARM, ear=0.12, fatigue_level=2)
    assert out is not frame
    assert np.array_equal(frame, np.zeros((120, 160, 3), dtype=np.uint8))


def test_draw_hud_actually_draws_something_when_alarming() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    out = draw_hud(frame, AlertTier.ALARM)
    assert out.sum() > 0  # some pixels were written


def test_draw_hud_renders_distraction_line() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    without = draw_hud(frame, AlertTier.NONE)
    with_distraction = draw_hud(
        frame,
        AlertTier.NONE,
        distraction_activity="texting_right",
        distraction_tier=AlertTier.VISUAL,
    )
    assert with_distraction.shape == frame.shape
    assert with_distraction.sum() > without.sum()  # the extra line drew pixels


def test_draw_hud_renders_gaze_line() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    without = draw_hud(frame, AlertTier.NONE)
    with_gaze = draw_hud(
        frame,
        AlertTier.NONE,
        gaze_zone="road_ahead",
        gaze_tier=AlertTier.NONE,
    )
    assert with_gaze.shape == frame.shape
    assert with_gaze.sum() > without.sum()


def test_draw_hud_gaze_line_differs_when_the_track_is_warning() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    on_road = draw_hud(frame, AlertTier.NONE, gaze_zone="road_ahead", gaze_tier=AlertTier.NONE)
    off_road = draw_hud(frame, AlertTier.NONE, gaze_zone="phone", gaze_tier=AlertTier.VISUAL)
    assert not np.array_equal(on_road, off_road)


def test_draw_hud_gaze_line_does_not_overwrite_the_distraction_line() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    distraction_only = draw_hud(
        frame,
        AlertTier.NONE,
        distraction_activity="texting_right",
        distraction_tier=AlertTier.VISUAL,
    )
    both = draw_hud(
        frame,
        AlertTier.NONE,
        distraction_activity="texting_right",
        distraction_tier=AlertTier.VISUAL,
        gaze_zone="phone",
        gaze_tier=AlertTier.VISUAL,
    )
    # The distraction row must survive untouched when the gaze row is added.
    assert np.array_equal(both[214:, :, :], distraction_only[214:, :, :])
    assert both.sum() > distraction_only.sum()


def test_draw_hud_gaze_args_are_optional() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    baseline = draw_hud(frame, AlertTier.NONE, ear=0.3, fatigue_level=0)
    unchanged = draw_hud(
        frame,
        AlertTier.NONE,
        ear=0.3,
        fatigue_level=0,
        gaze_zone=None,
        gaze_tier=None,
    )
    assert np.array_equal(baseline, unchanged)


def test_draw_hud_distraction_args_are_optional() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    baseline = draw_hud(frame, AlertTier.NONE, ear=0.3, fatigue_level=0)
    unchanged = draw_hud(
        frame,
        AlertTier.NONE,
        ear=0.3,
        fatigue_level=0,
        distraction_activity=None,
        distraction_tier=None,
    )
    assert np.array_equal(baseline, unchanged)
