import numpy as np
import pytest

from safeeyes.data.yawdd_crops import extract_clip_crops, extract_manifest_crops
from safeeyes.perception.landmarks import MOUTH_MAR_INDICES


def _face_landmarks(cx=320.0, cy=240.0, half=20.0, n=478):
    points = np.zeros((n, 3), dtype=float)
    offsets = [(-half, 0), (0, -half), (half, 0), (half, 0), (0, half), (-half, 0)]
    for index, (dx, dy) in zip(MOUTH_MAR_INDICES, offsets, strict=True):
        points[index] = (cx + dx, cy + dy, 0.0)
    return points


def _mar_seq(values):
    remaining = list(values)

    def to_features(landmarks, width, height):
        row = np.zeros(5, dtype=float)
        row[1] = remaining.pop(0)
        return row

    return to_features


class FakeDetector:
    def __init__(self, landmarks):
        self._landmarks = landmarks
        self.calls = 0

    def landmarks(self, frame):
        self.calls += 1
        return self._landmarks


class _MissingSecondFaceDetector:
    def __init__(self, landmarks):
        self._landmarks = landmarks
        self.calls = 0

    def landmarks(self, frame):
        self.calls += 1
        return None if self.calls == 2 else self._landmarks


def _detector_missing_second_face():
    return _MissingSecondFaceDetector(_face_landmarks())


def _frames(n):
    return [np.full((480, 640, 3), i % 255, dtype=np.uint8) for i in range(n)]


def test_frame_indices_are_real_video_indices():
    detector = FakeDetector(_face_landmarks())
    result = extract_clip_crops(
        _frames(9), detector, frame_step=3, to_features=_mar_seq([0.1, 0.9, 0.1])
    )
    assert result["frame_indices"].tolist() == [0, 3, 6]


def test_no_face_frames_shift_indices_but_not_alignment():
    detector = _detector_missing_second_face()
    result = extract_clip_crops(
        _frames(9), detector, frame_step=3, to_features=_mar_seq([0.1, 0.9])
    )
    assert result["features"].shape[0] == result["frame_indices"].shape[0]
    assert result["frame_indices"].tolist() == [0, 6]


def test_crops_are_written_only_near_gated_frames():
    detector = FakeDetector(_face_landmarks())
    result = extract_clip_crops(
        _frames(30),
        detector,
        frame_step=3,
        gate=0.45,
        margin_steps=1,
        to_features=_mar_seq([0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    )
    assert result["crop_rows"].tolist() == [1, 2, 3]


def test_crop_stack_shape_matches_crop_rows():
    detector = FakeDetector(_face_landmarks())
    result = extract_clip_crops(
        _frames(15), detector, frame_step=3, to_features=_mar_seq([0.9] * 5)
    )
    assert result["crops"].shape == (result["crop_rows"].size, 96, 96, 3)
    assert result["crops"].dtype == np.uint8


def test_crop_pixels_come_from_the_frame_their_row_names():
    # The strongest alignment check: every frame here is a flat field whose value
    # is its own video frame index, so a crop taken from the wrong moment carries
    # the wrong value and this fails. An off by one in either direction, or a
    # crop stack ordered independently of crop_rows, is caught here.
    detector = FakeDetector(_face_landmarks())
    result = extract_clip_crops(
        _frames(30),
        detector,
        frame_step=3,
        gate=0.45,
        margin_steps=1,
        to_features=_mar_seq([0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1]),
    )
    expected = result["frame_indices"][result["crop_rows"]]
    observed = np.array([int(crop[48, 48, 0]) for crop in result["crops"]])
    assert observed.tolist() == expected.tolist()
    assert result["crop_rows"].tolist() == [1, 2, 3, 6, 7, 8]


def test_margin_is_applied_on_both_sides_independently():
    # A gate hit at the very first row must still produce forward crops, and a
    # gate hit at the very last row must still produce backward crops. Clamping
    # to the sequence bounds must not silently drop the other side.
    detector = FakeDetector(_face_landmarks())
    first = extract_clip_crops(
        _frames(18),
        detector,
        frame_step=3,
        gate=0.45,
        margin_steps=2,
        to_features=_mar_seq([0.9, 0.1, 0.1, 0.1, 0.1, 0.1]),
    )
    assert first["crop_rows"].tolist() == [0, 1, 2]
    last = extract_clip_crops(
        _frames(18),
        FakeDetector(_face_landmarks()),
        frame_step=3,
        gate=0.45,
        margin_steps=2,
        to_features=_mar_seq([0.1, 0.1, 0.1, 0.1, 0.1, 0.9]),
    )
    assert last["crop_rows"].tolist() == [3, 4, 5]


def test_crop_rows_index_into_features_not_into_the_video():
    detector = _detector_missing_second_face()
    result = extract_clip_crops(
        _frames(12),
        detector,
        frame_step=3,
        gate=0.45,
        margin_steps=0,
        to_features=_mar_seq([0.1, 0.9, 0.1]),
    )
    # Frame 3 has no face, so row 1 is video frame 6, and the gated row is 1.
    assert result["frame_indices"].tolist() == [0, 6, 9]
    assert result["crop_rows"].tolist() == [1]
    assert result["features"][result["crop_rows"][0], 1] == pytest.approx(0.9)
    assert int(result["crops"][0][48, 48, 0]) == 6


def test_no_gate_hits_yields_an_empty_but_well_shaped_crop_stack():
    detector = FakeDetector(_face_landmarks())
    result = extract_clip_crops(
        _frames(9), detector, frame_step=3, gate=0.45, to_features=_mar_seq([0.1, 0.2, 0.3])
    )
    assert result["features"].shape == (3, 5)
    assert result["crop_rows"].shape == (0,)
    assert result["crops"].shape == (0, 96, 96, 3)
    assert result["crops"].dtype == np.uint8


def test_a_clip_with_no_detected_faces_yields_empty_arrays():
    class _NoFace:
        def landmarks(self, frame):
            return None

    result = extract_clip_crops(_frames(9), _NoFace(), frame_step=3, to_features=_mar_seq([]))
    assert result["features"].shape == (0, 5)
    assert result["frame_indices"].shape == (0,)
    assert result["crop_rows"].shape == (0,)
    assert result["crops"].shape == (0, 96, 96, 3)


def test_crop_size_is_honoured():
    detector = FakeDetector(_face_landmarks())
    result = extract_clip_crops(
        _frames(9), detector, frame_step=3, size=64, to_features=_mar_seq([0.9, 0.9, 0.9])
    )
    assert result["crops"].shape == (3, 64, 64, 3)


def _write_manifest(path, sample_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["sample_id,subject_id,label"]
    lines.extend(f"{sample_id},s1,yawning" for sample_id in sample_ids)
    path.write_text("\n".join(lines) + "\n")


def test_manifest_extraction_writes_one_npz_per_sample(tmp_path, monkeypatch):
    import safeeyes.data.yawdd_crops as module

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, ["a.avi", "b.avi"])
    monkeypatch.setattr(module, "iter_video_frames", lambda path: _frames(9))
    written = extract_manifest_crops(
        [manifest],
        tmp_path / "videos",
        tmp_path / "out",
        FakeDetector(_face_landmarks()),
        frame_step=3,
        to_features=_mar_seq([0.9] * 6),
    )
    assert [p.name for p in written] == ["a.npz", "b.npz"]
    loaded = np.load(written[0])
    assert set(loaded.files) == {"features", "frame_indices", "crop_rows", "crops"}
    assert loaded["frame_indices"].tolist() == [0, 3, 6]


def test_manifest_extraction_skips_existing_by_default(tmp_path, monkeypatch):
    import safeeyes.data.yawdd_crops as module

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, ["a.avi"])
    monkeypatch.setattr(module, "iter_video_frames", lambda path: _frames(9))
    detector = FakeDetector(_face_landmarks())
    extract_manifest_crops(
        [manifest],
        tmp_path / "videos",
        tmp_path / "out",
        detector,
        frame_step=3,
        to_features=_mar_seq([0.9] * 3),
    )
    after_first = detector.calls
    extract_manifest_crops(
        [manifest],
        tmp_path / "videos",
        tmp_path / "out",
        detector,
        frame_step=3,
        to_features=_mar_seq([0.9] * 3),
    )
    assert detector.calls == after_first


def test_manifest_extraction_honours_limit(tmp_path, monkeypatch):
    import safeeyes.data.yawdd_crops as module

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, ["a.avi", "b.avi", "c.avi"])
    monkeypatch.setattr(module, "iter_video_frames", lambda path: _frames(9))
    written = extract_manifest_crops(
        [manifest],
        tmp_path / "videos",
        tmp_path / "out",
        FakeDetector(_face_landmarks()),
        frame_step=3,
        limit=2,
        to_features=_mar_seq([0.9] * 6),
    )
    assert [p.name for p in written] == ["a.npz", "b.npz"]
