from pathlib import Path

import numpy as np
import pytest

from safeeyes.data.manifest import write_manifest
from safeeyes.data.splits import Sample
from safeeyes.perception.extract import (
    FEATURE_DIM,
    extract_clip_features,
    extract_dataset_features,
)


class _StubDetector:
    """Returns a preset landmark result per frame, mimicking FaceMeshDetector."""

    def __init__(self, results: list[np.ndarray | None]) -> None:
        self._results = results
        self._index = 0

    def landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        result = self._results[self._index]
        self._index += 1
        return result


def _frame() -> np.ndarray:
    return np.zeros((8, 12, 3), dtype=np.uint8)


def test_extract_clip_skips_frames_without_a_face() -> None:
    frames = [_frame(), _frame(), _frame()]
    detector = _StubDetector([np.array([[1.0, 0.0]]), None, np.array([[3.0, 0.0]])])
    # Inject a feature function so the test exercises control flow, not landmark math.
    out = extract_clip_features(
        frames, detector, to_features=lambda lm, w, h: np.full(FEATURE_DIM, lm[0, 0])
    )
    assert out.shape == (2, FEATURE_DIM)
    assert out[0, 0] == 1.0
    assert out[1, 0] == 3.0


def test_extract_clip_passes_frame_dimensions_to_features() -> None:
    frames = [_frame()]  # shape (8, 12, 3) -> height 8, width 12
    detector = _StubDetector([np.array([[0.0, 0.0]])])
    seen: dict[str, int] = {}

    def spy(lm: np.ndarray, w: int, h: int) -> np.ndarray:
        seen["w"], seen["h"] = w, h
        return np.zeros(FEATURE_DIM)

    extract_clip_features(frames, detector, to_features=spy)
    assert seen == {"w": 12, "h": 8}


def test_extract_clip_with_no_faces_returns_empty_feature_array() -> None:
    frames = [_frame(), _frame()]
    detector = _StubDetector([None, None])
    out = extract_clip_features(
        frames, detector, to_features=lambda lm, w, h: np.zeros(FEATURE_DIM)
    )
    assert out.shape == (0, FEATURE_DIM)


def test_extract_clip_frame_step_subsamples() -> None:
    frames = [_frame(), _frame(), _frame(), _frame()]
    detector = _StubDetector([np.array([[float(i), 0.0]]) for i in range(4)])
    out = extract_clip_features(
        frames, detector, to_features=lambda lm, w, h: np.full(FEATURE_DIM, lm[0, 0]), frame_step=2
    )
    # Two of four frames are processed; the detector is consulted only on those,
    # so it returns its first two results in order.
    assert out.shape == (2, FEATURE_DIM)
    assert out[0, 0] == 0.0
    assert out[1, 0] == 1.0


def _uta_manifest(tmp_path: Path) -> Path:
    samples = [
        Sample(sample_id="Fold1/s01/0.mp4", subject_id="s01", label="alert"),
        Sample(sample_id="Fold1/s02/10.mp4", subject_id="s02", label="drowsy"),
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(samples, manifest)
    return manifest


def test_extract_dataset_writes_features_mirroring_sample_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _uta_manifest(tmp_path)
    seen: list[str] = []

    def fake_extract(path: object, detector: object, **kwargs: object) -> np.ndarray:
        seen.append(str(path))
        return np.full((4, FEATURE_DIM), float(len(seen)))

    monkeypatch.setattr("safeeyes.perception.extract.extract_video_features", fake_extract)
    written = extract_dataset_features(
        [manifest], tmp_path / "vids", tmp_path / "feats", detector=object()
    )

    assert (tmp_path / "feats/Fold1/s01/0.npy").exists()
    assert (tmp_path / "feats/Fold1/s02/10.npy").exists()
    assert written == [tmp_path / "feats/Fold1/s01/0.npy", tmp_path / "feats/Fold1/s02/10.npy"]
    # the per-video extractor was pointed at the video-root joined sample id
    assert seen[0].endswith("vids/Fold1/s01/0.mp4")
    assert np.load(tmp_path / "feats/Fold1/s01/0.npy").shape == (4, FEATURE_DIM)


def test_extract_dataset_skips_already_extracted_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _uta_manifest(tmp_path)
    feats = tmp_path / "feats"
    existing = feats / "Fold1/s01/0.npy"
    existing.parent.mkdir(parents=True)
    np.save(existing, np.zeros((3, FEATURE_DIM)))

    calls: list[str] = []

    def fake_extract(path: object, detector: object, **kwargs: object) -> np.ndarray:
        calls.append(str(path))
        return np.full((1, FEATURE_DIM), 9.0)

    monkeypatch.setattr("safeeyes.perception.extract.extract_video_features", fake_extract)
    extract_dataset_features(
        [manifest], tmp_path / "vids", feats, detector=object(), skip_existing=True
    )

    # only the missing clip is extracted; the existing one is left untouched
    assert len(calls) == 1
    assert calls[0].endswith("Fold1/s02/10.mp4")
    assert np.load(existing).shape == (3, FEATURE_DIM)
