"""Assemble labeled mouth opening events from mar sequences.

YawDD ships no per frame annotations, only a label on each video's file
name. A video labeled Yawning contains one or more yawns plus a great deal
of ordinary mouth movement, so treating every detected mouth opening in that
video as a yawn would teach a classifier the exact confusion this dataset is
meant to remove. The rule applied here is therefore conservative: in a video
whose label contains Yawning, only the single longest event is kept as a
positive and every other event in that video is dropped from training. In a
Normal or Talking video every detected event is a negative, since none of
those labels claim a yawn occurred.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from safeeyes.data.yawdd import is_yawning
from safeeyes.temporal.yawn_validation import event_runs


@dataclass(frozen=True)
class YawnEvent:
    sample_id: str
    subject_id: str
    start: int
    end: int
    peak_mar: float
    label: int


def events_for_video(
    sample_id: str,
    subject_id: str,
    video_label: str,
    mar: np.ndarray,
    threshold: float,
    min_duration: int = 1,
) -> list[YawnEvent]:
    values = np.asarray(mar, dtype=float)
    runs = [
        (start, end)
        for start, end in event_runs(values, threshold)
        if end - start + 1 >= min_duration
    ]
    if not runs:
        return []

    yawning = is_yawning(video_label.split("&"))

    candidates = [(start, end, float(values[start : end + 1].max())) for start, end in runs]

    if not yawning:
        return [
            YawnEvent(
                sample_id=sample_id,
                subject_id=subject_id,
                start=start,
                end=end,
                peak_mar=peak_mar,
                label=0,
            )
            for start, end, peak_mar in candidates
        ]

    start, end, peak_mar = min(candidates, key=lambda c: (-(c[1] - c[0]), -c[2], c[0]))
    return [
        YawnEvent(
            sample_id=sample_id,
            subject_id=subject_id,
            start=start,
            end=end,
            peak_mar=peak_mar,
            label=1,
        )
    ]


def training_events(
    videos: Sequence[tuple[str, str, str, np.ndarray]],
    threshold: float,
    min_duration: int = 1,
) -> list[YawnEvent]:
    events: list[YawnEvent] = []
    for sample_id, subject_id, video_label, mar in videos:
        events.extend(
            events_for_video(sample_id, subject_id, video_label, mar, threshold, min_duration)
        )
    return events
