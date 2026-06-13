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
