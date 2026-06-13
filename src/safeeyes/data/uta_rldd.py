"""UTA-RLDD manifest building.

The Real Life Drowsiness Dataset ships as folds of per subject folders, each
holding one video per drowsiness class. The class is encoded in the video file
name: 0 (alert), 5 (low vigilance), 10 (drowsy). The subject is the parent
folder, which is what the subject independent split keys on.

Each manifest entry is one video. Frame and window extraction happen downstream
in preprocessing; the split is decided at the subject level before any frame is
touched, so no subject can leak across train and test.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from safeeyes.data.splits import Sample

UTA_LABELS = {"0": "alert", "5": "low_vigilance", "10": "drowsy"}
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv"})


def uta_label_from_stem(stem: str) -> str:
    try:
        return UTA_LABELS[stem]
    except KeyError:
        raise ValueError(
            f"unknown UTA-RLDD class stem {stem!r}; expected one of {sorted(UTA_LABELS)}"
        ) from None


def build_uta_manifest(
    root: str | Path,
    video_extensions: Iterable[str] = VIDEO_EXTENSIONS,
) -> list[Sample]:
    root = Path(root)
    exts = {e.lower() for e in video_extensions}
    samples: list[Sample] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        if path.stem not in UTA_LABELS:
            continue
        samples.append(
            Sample(
                sample_id=str(path.relative_to(root)),
                subject_id=path.parent.name,
                label=uta_label_from_stem(path.stem),
            )
        )
    return samples
