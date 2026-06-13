import pytest

from safeeyes.data.frames import extract_frames, sample_frame_indices


def test_downsamples_to_target_fps() -> None:
    # 30 fps source, want 10 fps: every third frame.
    indices = sample_frame_indices(total_frames=30, source_fps=30.0, target_fps=10.0)
    assert indices == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]


def test_target_at_or_above_source_keeps_every_frame() -> None:
    assert sample_frame_indices(total_frames=5, source_fps=30.0, target_fps=30.0) == [0, 1, 2, 3, 4]
    assert sample_frame_indices(total_frames=5, source_fps=30.0, target_fps=60.0) == [0, 1, 2, 3, 4]


def test_indices_are_increasing_and_in_range() -> None:
    indices = sample_frame_indices(total_frames=1000, source_fps=29.97, target_fps=5.0)
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)
    assert all(0 <= i < 1000 for i in indices)


def test_zero_frames_yields_no_indices() -> None:
    assert sample_frame_indices(total_frames=0, source_fps=30.0, target_fps=10.0) == []


def test_non_positive_fps_raises() -> None:
    with pytest.raises(ValueError):
        sample_frame_indices(total_frames=10, source_fps=0.0, target_fps=10.0)
    with pytest.raises(ValueError):
        sample_frame_indices(total_frames=10, source_fps=30.0, target_fps=-1.0)


def test_extract_frames_reads_only_requested_indices_in_order() -> None:
    reads: list[int] = []

    def fake_reader(i: int) -> str:
        reads.append(i)
        return f"frame{i}"

    out = extract_frames(fake_reader, [0, 3, 6])
    assert out == ["frame0", "frame3", "frame6"]
    assert reads == [0, 3, 6]
