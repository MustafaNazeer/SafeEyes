"""Alert level metrics over replayed clips.

A false alarm is an excursion of the alert tier to at or above a threshold tier
during a clip whose label says the driver was not drowsy. Metrics are reported
per threshold tier so the headline (AUDIBLE or above) sits beside the stricter
and looser definitions. Low vigilance clips count as not drowsy, mirroring the
committed window level false alarm definition; the variant restricted to alert
labeled clips is reported alongside.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from safeeyes.alert.replay import TierEvent
from safeeyes.alert.state_machine import AlertTier

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
