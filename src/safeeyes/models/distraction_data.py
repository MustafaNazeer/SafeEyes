"""Frame level dataset for the DMD distraction classifier.

Interval manifests name video segments; the frame extractor has already turned
each segment into a directory of jpgs. This module resolves those directories,
flattens manifests into per frame records, and wraps the records in a torch
Dataset. The image transform is injected by the training CLI, so this module
stays pure path and record logic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from safeeyes.data.dmd_distraction import DMD_DISTRACTION_ACTIONS
from safeeyes.data.intervals import read_interval_manifest, sanitize_sample_id

DISTRACTION_LABELS: tuple[str, ...] = tuple(
    sorted(action.removeprefix("driver_actions/") for action in DMD_DISTRACTION_ACTIONS)
)
LABEL_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(DISTRACTION_LABELS)}


def interval_frame_paths(frames_root: str | Path, sample_id: str) -> list[Path]:
    directory = Path(frames_root) / sanitize_sample_id(sample_id)
    if not directory.is_dir():
        raise FileNotFoundError(f"frame directory not found: {directory}")
    frames = sorted(directory.glob("frame_*.jpg"))
    if not frames:
        raise ValueError(f"frame directory is empty: {directory}")
    return frames


@dataclass(frozen=True)
class FrameRecord:
    path: Path
    label_index: int
    sample_id: str


def frame_records(
    manifest_paths: Sequence[str | Path], frames_root: str | Path
) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for manifest_path in manifest_paths:
        for sample in read_interval_manifest(manifest_path):
            if sample.label not in LABEL_TO_INDEX:
                raise ValueError(f"unknown distraction label {sample.label!r} in {manifest_path}")
            label_index = LABEL_TO_INDEX[sample.label]
            for path in interval_frame_paths(frames_root, sample.sample_id):
                records.append(
                    FrameRecord(path=path, label_index=label_index, sample_id=sample.sample_id)
                )
    return records


class FrameDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        records: Sequence[FrameRecord],
        transform: Callable[[np.ndarray], torch.Tensor],
    ) -> None:
        self._records = list(records)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self._records[index]
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read image: {record.path}")
        return self._transform(image), record.label_index
