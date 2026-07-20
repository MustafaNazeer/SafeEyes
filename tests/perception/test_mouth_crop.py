import numpy as np

from safeeyes.perception.mouth_crop import crop_mouth, mouth_crop_box


def _landmarks_at(cx, cy, half=20.0, n=478):
    points = np.zeros((n, 3), dtype=float)
    from safeeyes.perception.landmarks import MOUTH_MAR_INDICES

    offsets = [(-half, 0), (0, -half), (half, 0), (half, 0), (0, half), (-half, 0)]
    for index, (dx, dy) in zip(MOUTH_MAR_INDICES, offsets, strict=True):
        points[index] = (cx + dx, cy + dy, 0.0)
    return points


def test_box_is_square_away_from_the_frame_edge():
    x0, y0, x1, y1 = mouth_crop_box(_landmarks_at(320, 240), 640, 480)
    assert (x1 - x0) == (y1 - y0)


def _landmarks_with_unequal_spread(cx, cy, half_x, half_y, n=478):
    points = np.zeros((n, 3), dtype=float)
    from safeeyes.perception.landmarks import MOUTH_MAR_INDICES

    offsets = [
        (-half_x, 0),
        (0, -half_y),
        (half_x, 0),
        (half_x, 0),
        (0, half_y),
        (-half_x, 0),
    ]
    for index, (dx, dy) in zip(MOUTH_MAR_INDICES, offsets, strict=True):
        points[index] = (cx + dx, cy + dy, 0.0)
    return points


def test_box_is_square_with_unequal_horizontal_and_vertical_spread():
    x0, y0, x1, y1 = mouth_crop_box(
        _landmarks_with_unequal_spread(320, 240, half_x=30.0, half_y=10.0), 640, 480
    )
    assert (x1 - x0) == (y1 - y0)


def test_box_is_centered_on_the_mouth():
    x0, y0, x1, y1 = mouth_crop_box(_landmarks_at(320, 240), 640, 480)
    assert abs((x0 + x1) / 2 - 320) <= 1
    assert abs((y0 + y1) / 2 - 240) <= 1


def test_margin_widens_the_box():
    narrow = mouth_crop_box(_landmarks_at(320, 240), 640, 480, margin=0.0)
    wide = mouth_crop_box(_landmarks_at(320, 240), 640, 480, margin=0.5)
    assert (wide[2] - wide[0]) > (narrow[2] - narrow[0])


def test_box_is_clamped_to_the_frame():
    x0, y0, x1, y1 = mouth_crop_box(_landmarks_at(5, 5), 640, 480)
    assert x0 >= 0 and y0 >= 0 and x1 <= 640 and y1 <= 480


def test_crop_returns_the_requested_size():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = crop_mouth(frame, _landmarks_at(320, 240), size=96)
    assert crop.shape == (96, 96, 3)
    assert crop.dtype == np.uint8


def test_crop_contains_the_mouth_pixels():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[230:250, 310:330] = 255
    crop = crop_mouth(frame, _landmarks_at(320, 240), size=96)
    assert crop.max() == 255
