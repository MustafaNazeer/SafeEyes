from __future__ import annotations

import pytest

from safeeyes.data.interval_frames import interval_frame_indices, sanitize_sample_id


def test_indices_cover_interval_at_stride() -> None:
    assert interval_frame_indices(10, 19, stride=3, max_frames=None) == [10, 13, 16, 19]


def test_single_frame_interval_yields_one_index() -> None:
    assert interval_frame_indices(5, 5, stride=10, max_frames=None) == [5]


def test_max_frames_caps_evenly() -> None:
    indices = interval_frame_indices(0, 99, stride=1, max_frames=5)
    assert len(indices) == 5
    assert indices[0] == 0 and indices[-1] == 99
    gaps = {b - a for a, b in zip(indices, indices[1:], strict=False)}
    assert max(gaps) - min(gaps) <= 1


def test_invalid_interval_raises() -> None:
    with pytest.raises(ValueError, match="interval"):
        interval_frame_indices(10, 9, stride=1, max_frames=None)


def test_invalid_stride_raises() -> None:
    with pytest.raises(ValueError, match="stride"):
        interval_frame_indices(0, 10, stride=0, max_frames=None)


def test_sanitized_sample_id_is_a_safe_relative_path() -> None:
    sid = "gA/1/s1/gA_1_s1_2019-01-01T00;00;00+01;00_rgb_body.mp4#10-40"
    out = sanitize_sample_id(sid)
    assert "#" not in out
    assert out == "gA/1/s1/gA_1_s1_2019-01-01T00;00;00+01;00_rgb_body.mp4_10-40"
