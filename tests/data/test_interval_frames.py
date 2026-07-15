from __future__ import annotations

from pathlib import Path

import pytest

from safeeyes.data.interval_frames import (
    extract_manifest_frames,
    interval_frame_indices,
    sanitize_sample_id,
)
from safeeyes.data.intervals import IntervalSample, write_interval_manifest


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


def test_max_frames_one_returns_single_index_within_interval() -> None:
    indices = interval_frame_indices(0, 10, stride=1, max_frames=1)
    assert len(indices) == 1
    assert 0 <= indices[0] <= 10


def test_invalid_max_frames_raises() -> None:
    with pytest.raises(ValueError, match="max_frames"):
        interval_frame_indices(0, 10, stride=1, max_frames=0)


def test_sanitized_sample_id_is_a_safe_relative_path() -> None:
    sid = "gA/1/s1/gA_1_s1_2019-01-01T00;00;00+01;00_rgb_body.mp4#10-40"
    out = sanitize_sample_id(sid)
    assert "#" not in out
    assert out == "gA/1/s1/gA_1_s1_2019-01-01T00;00;00+01;00_rgb_body.mp4_10-40"


def _write_synthetic_video(path: Path, frame_count: int) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32)
    )
    try:
        for i in range(frame_count):
            writer.write(np.full((32, 32, 3), i % 256, dtype=np.uint8))
    finally:
        writer.release()


def _write_manifest(path: Path, samples: list[IntervalSample]) -> None:
    write_interval_manifest(samples, path)


def test_tail_overshoot_is_clamped_to_video_end(tmp_path: Path) -> None:
    _write_synthetic_video(tmp_path / "videos" / "clip.mp4", frame_count=10)
    manifest = tmp_path / "train.csv"
    _write_manifest(
        manifest,
        [IntervalSample("clip.mp4#0-19", "s1", "drinking", 0, 19)],
    )
    truncated: list[tuple[str, int]] = []
    written = extract_manifest_frames(
        [manifest],
        tmp_path / "videos",
        tmp_path / "out",
        stride=1,
        max_frames=None,
        on_truncated=lambda sid, dropped: truncated.append((sid, dropped)),
    )
    assert len(written) == 10
    assert truncated == [("clip.mp4#0-19", 10)]
    rerun_truncated: list[tuple[str, int]] = []
    rerun = extract_manifest_frames(
        [manifest],
        tmp_path / "videos",
        tmp_path / "out",
        stride=1,
        max_frames=None,
        on_truncated=lambda sid, dropped: rerun_truncated.append((sid, dropped)),
    )
    assert rerun == []
    assert rerun_truncated == [("clip.mp4#0-19", 10)]


def test_sample_entirely_past_video_end_raises(tmp_path: Path) -> None:
    _write_synthetic_video(tmp_path / "videos" / "clip.mp4", frame_count=10)
    manifest = tmp_path / "train.csv"
    _write_manifest(
        manifest,
        [IntervalSample("clip.mp4#15-19", "s1", "drinking", 15, 19)],
    )
    with pytest.raises(ValueError, match="past the end"):
        extract_manifest_frames(
            [manifest], tmp_path / "videos", tmp_path / "out", stride=1, max_frames=None
        )


def test_decode_falling_short_of_metadata_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2

    _write_synthetic_video(tmp_path / "videos" / "clip.mp4", frame_count=10)
    real_capture = cv2.VideoCapture

    class InflatedCapture:
        def __init__(self, path: str) -> None:
            self._inner = real_capture(path)

        def get(self, prop: int) -> float:
            value = self._inner.get(prop)
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return value + 5
            return value

        def read(self):  # type: ignore[no-untyped-def]
            return self._inner.read()

        def release(self) -> None:
            self._inner.release()

    monkeypatch.setattr(cv2, "VideoCapture", InflatedCapture)
    manifest = tmp_path / "train.csv"
    _write_manifest(
        manifest,
        [IntervalSample("clip.mp4#0-14", "s1", "drinking", 0, 14)],
    )
    with pytest.raises(ValueError, match="unwritten"):
        extract_manifest_frames(
            [manifest], tmp_path / "videos", tmp_path / "out", stride=1, max_frames=None
        )


def test_duplicate_sample_id_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "train.csv"
    _write_manifest(
        manifest,
        [
            IntervalSample("clip.mp4#0-5", "s1", "drinking", 0, 5),
            IntervalSample("clip.mp4#0-5", "s1", "radio", 0, 5),
        ],
    )
    with pytest.raises(ValueError, match="duplicate sample id"):
        extract_manifest_frames([manifest], tmp_path / "videos", tmp_path / "out")


def test_extracts_requested_frames_and_resumes(tmp_path: Path) -> None:
    _write_synthetic_video(tmp_path / "videos" / "clip.mp4", frame_count=30)
    manifest = tmp_path / "train.csv"
    _write_manifest(
        manifest,
        [
            IntervalSample("clip.mp4#0-9", "s1", "drinking", 0, 9),
            IntervalSample("clip.mp4#20-29", "s1", "radio", 20, 29),
        ],
    )
    written = extract_manifest_frames(
        [manifest], tmp_path / "videos", tmp_path / "out", stride=5, max_frames=None
    )
    assert sorted(p.name for p in written if "0-9" in str(p.parent)) == [
        "frame_000000.jpg",
        "frame_000005.jpg",
        "frame_000009.jpg",
    ]
    assert len(written) == 6
    rerun = extract_manifest_frames(
        [manifest], tmp_path / "videos", tmp_path / "out", stride=5, max_frames=None
    )
    assert rerun == []
