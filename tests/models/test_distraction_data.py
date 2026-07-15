from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from safeeyes.data.dmd_distraction import DMD_DISTRACTION_ACTIONS
from safeeyes.data.intervals import IntervalSample, sanitize_sample_id, write_interval_manifest
from safeeyes.models.distraction_data import (
    DISTRACTION_LABELS,
    LABEL_TO_INDEX,
    FrameDataset,
    FrameRecord,
    frame_records,
    interval_frame_paths,
)

EXPECTED_LABELS = (
    "change_gear",
    "drinking",
    "hair_and_makeup",
    "phonecall_left",
    "phonecall_right",
    "radio",
    "reach_backseat",
    "reach_side",
    "safe_drive",
    "standstill_or_waiting",
    "talking_to_passenger",
    "texting_left",
    "texting_right",
)


def _write_frames(frames_root: Path, sample_id: str, indices: list[int]) -> list[Path]:
    directory = frames_root / sanitize_sample_id(sample_id)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in indices:
        path = directory / f"frame_{index:06d}.jpg"
        cv2.imwrite(str(path), np.full((8, 8, 3), 128, dtype=np.uint8))
        paths.append(path)
    return paths


def test_distraction_labels_are_the_pinned_thirteen() -> None:
    assert DISTRACTION_LABELS == EXPECTED_LABELS


def test_distraction_labels_derive_from_dmd_actions() -> None:
    derived = tuple(
        sorted(action.removeprefix("driver_actions/") for action in DMD_DISTRACTION_ACTIONS)
    )
    assert DISTRACTION_LABELS == derived


def test_label_to_index_matches_tuple_positions() -> None:
    assert LABEL_TO_INDEX == {label: i for i, label in enumerate(DISTRACTION_LABELS)}


def test_interval_frame_paths_returns_sorted_frames(tmp_path: Path) -> None:
    sample_id = "gA/1/clip_rgb_body.mp4#10-40"
    _write_frames(tmp_path, sample_id, [40, 10, 25])
    paths = interval_frame_paths(tmp_path, sample_id)
    assert [p.name for p in paths] == ["frame_000010.jpg", "frame_000025.jpg", "frame_000040.jpg"]
    assert all(p.parent == tmp_path / sanitize_sample_id(sample_id) for p in paths)


def test_interval_frame_paths_missing_directory_raises(tmp_path: Path) -> None:
    sample_id = "clip_rgb_body.mp4#0-5"
    expected_dir = tmp_path / sanitize_sample_id(sample_id)
    with pytest.raises(FileNotFoundError, match=str(expected_dir)):
        interval_frame_paths(tmp_path, sample_id)


def test_interval_frame_paths_empty_directory_raises(tmp_path: Path) -> None:
    sample_id = "clip_rgb_body.mp4#0-5"
    (tmp_path / sanitize_sample_id(sample_id)).mkdir(parents=True)
    with pytest.raises(ValueError):
        interval_frame_paths(tmp_path, sample_id)


def test_frame_records_flattens_manifest_samples(tmp_path: Path) -> None:
    frames_root = tmp_path / "frames"
    samples = [
        IntervalSample(
            sample_id="a_rgb_body.mp4#0-10",
            subject_id="gA_1",
            label="safe_drive",
            start_frame=0,
            end_frame=10,
        ),
        IntervalSample(
            sample_id="b_rgb_body.mp4#5-9",
            subject_id="gA_2",
            label="drinking",
            start_frame=5,
            end_frame=9,
        ),
    ]
    _write_frames(frames_root, samples[0].sample_id, [0, 5, 10])
    _write_frames(frames_root, samples[1].sample_id, [5, 9])
    manifest = tmp_path / "train.csv"
    write_interval_manifest(samples, manifest)

    records = frame_records([manifest], frames_root)

    assert len(records) == 5
    assert [r.sample_id for r in records] == [samples[0].sample_id] * 3 + [
        samples[1].sample_id
    ] * 2
    assert [r.label_index for r in records[:3]] == [LABEL_TO_INDEX["safe_drive"]] * 3
    assert [r.label_index for r in records[3:]] == [LABEL_TO_INDEX["drinking"]] * 2
    assert all(r.path.is_file() for r in records)


def test_frame_records_unknown_label_raises(tmp_path: Path) -> None:
    frames_root = tmp_path / "frames"
    sample = IntervalSample(
        sample_id="a_rgb_body.mp4#0-10",
        subject_id="gA_1",
        label="juggling",
        start_frame=0,
        end_frame=10,
    )
    _write_frames(frames_root, sample.sample_id, [0])
    manifest = tmp_path / "train.csv"
    write_interval_manifest([sample], manifest)
    with pytest.raises(ValueError, match="juggling"):
        frame_records([manifest], frames_root)


def test_frame_dataset_returns_transformed_tensor_and_label(tmp_path: Path) -> None:
    sample_id = "a_rgb_body.mp4#0-10"
    (path,) = _write_frames(tmp_path, sample_id, [0])
    record = FrameRecord(path=path, label_index=4, sample_id=sample_id)

    def transform(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)

    dataset = FrameDataset([record], transform)
    assert len(dataset) == 1
    tensor, label = dataset[0]
    assert tensor.shape == (3, 8, 8)
    assert tensor.dtype == torch.float32
    assert label == 4


def test_frame_dataset_missing_image_raises(tmp_path: Path) -> None:
    record = FrameRecord(
        path=tmp_path / "gone.jpg", label_index=0, sample_id="a_rgb_body.mp4#0-10"
    )
    dataset = FrameDataset([record], lambda image: torch.from_numpy(image))
    with pytest.raises(FileNotFoundError):
        dataset[0]
