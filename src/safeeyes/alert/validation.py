"""Alert level metrics over replayed clips.

A false alarm is an excursion of the alert tier to at or above a threshold tier
during a clip whose label says the driver was not drowsy. Metrics are reported
per threshold tier so the headline (AUDIBLE or above) sits beside the stricter
and looser definitions. Low vigilance clips count as not drowsy, mirroring the
committed window level false alarm definition; the variant restricted to alert
labeled clips is reported alongside.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from safeeyes.alert.replay import TierEvent, replay_levels
from safeeyes.alert.state_machine import AlertStateMachine, AlertTier

NOT_DROWSY_LABELS = ("alert", "low_vigilance")
DROWSY_LABEL = "drowsy"


@dataclass(frozen=True)
class ClipReplay:
    sample_id: str
    label: str
    n_frames: int
    events: list[TierEvent]


def count_alarms(events: Sequence[TierEvent], threshold: AlertTier) -> int:
    count = 0
    prev = AlertTier.NONE
    for event in events:
        if event.tier >= threshold and prev < threshold:
            count += 1
        prev = event.tier
    return count


def first_alarm_frame(events: Sequence[TierEvent], threshold: AlertTier) -> int | None:
    prev = AlertTier.NONE
    for event in events:
        if event.tier >= threshold and prev < threshold:
            return event.frame
        prev = event.tier
    return None


def _hours(clips: Sequence[ClipReplay], fps: float) -> float:
    return sum(c.n_frames for c in clips) / fps / 3600.0


def summarize_replays(
    clips: Sequence[ClipReplay], threshold: AlertTier, fps: float
) -> dict[str, object]:
    for clip in clips:
        if clip.label != DROWSY_LABEL and clip.label not in NOT_DROWSY_LABELS:
            raise ValueError(f"unknown clip label: {clip.label!r}")
    not_drowsy = [c for c in clips if c.label in NOT_DROWSY_LABELS]
    alert_only = [c for c in clips if c.label == "alert"]
    drowsy = [c for c in clips if c.label == DROWSY_LABEL]

    nd_alarms = sum(count_alarms(c.events, threshold) for c in not_drowsy)
    ao_alarms = sum(count_alarms(c.events, threshold) for c in alert_only)
    detections = [
        f
        for c in drowsy
        if (f := first_alarm_frame(c.events, threshold)) is not None
    ]
    return {
        "n_clips": len(clips),
        "n_not_drowsy_clips": len(not_drowsy),
        "n_drowsy_clips": len(drowsy),
        "false_alarms_per_hour": (
            nd_alarms / _hours(not_drowsy, fps) if not_drowsy else None
        ),
        "fraction_not_drowsy_clips_with_alarm": (
            sum(1 for c in not_drowsy if count_alarms(c.events, threshold) > 0)
            / len(not_drowsy)
            if not_drowsy
            else None
        ),
        "false_alarms_per_hour_alert_clips_only": (
            ao_alarms / _hours(alert_only, fps) if alert_only else None
        ),
        "drowsy_detection_rate": (len(detections) / len(drowsy)) if drowsy else None,
        "median_time_to_first_alert_s": (
            statistics.median(f / fps for f in detections) if detections else None
        ),
    }


DEFAULT_PARAMS = (5, 15, 45)

SWEEP_GRID = [
    (e, d, a)
    for e, d, a in itertools.product((3, 5, 8, 12), (8, 15, 25, 40), (15, 30, 45, 90))
]


def sweep_parameters(
    level_sequences: Sequence[tuple[str, str, Sequence[int]]],
    grid: Sequence[tuple[int, int, int]],
    threshold: AlertTier,
    fps: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for escalate, de_escalate, alarm_after in grid:
        machine = AlertStateMachine(
            escalate_steps=escalate,
            de_escalate_steps=de_escalate,
            alarm_after=alarm_after,
        )
        clips = [
            ClipReplay(sample_id, label, len(levels), replay_levels(levels, machine))
            for sample_id, label, levels in level_sequences
        ]
        summary = summarize_replays(clips, threshold, fps)
        rows.append({"params": [escalate, de_escalate, alarm_after], **summary})
    return rows


def select_parameters(
    rows: Sequence[dict[str, object]], default: tuple[int, int, int] = DEFAULT_PARAMS
) -> dict[str, object]:
    """The pre-registered rule: among rows whose drowsy detection rate is at
    least the default parameters' rate, lowest false alarms per hour wins; ties
    break on faster median time to first alert. Fixed before any
    sweep ran, so the choice cannot be fished post hoc.
    """
    baseline = next(
        (r for r in rows if tuple(cast(list[int], r["params"])) == tuple(default)), None
    )
    if baseline is None:
        raise ValueError("sweep rows must include the default parameters as baseline")
    if baseline["drowsy_detection_rate"] is None:
        raise ValueError("baseline row has no drowsy clips to anchor the detection rate")
    baseline_detection = cast(float, baseline["drowsy_detection_rate"])
    eligible: list[dict[str, object]] = [
        r
        for r in rows
        if r["drowsy_detection_rate"] is not None
        and cast(float, r["drowsy_detection_rate"]) >= baseline_detection
    ]

    def sort_key(row: dict[str, object]) -> tuple[float, float]:
        median = cast(float | None, row["median_time_to_first_alert_s"])
        return (
            cast(float, row["false_alarms_per_hour"]),
            median if median is not None else math.inf,
        )

    return min(eligible, key=sort_key)
