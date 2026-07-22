"""On screen heads up overlay for the live demo.

Draws a colored status banner and a short label for the current alert tier, plus
optional readouts. This is a demo cue for a prototype, not a certified display.
The drawing is deliberately simple; the alert decision it visualizes is made by
the tested state machine.
"""

from __future__ import annotations

import cv2
import numpy as np

from safeeyes.alert.state_machine import AlertTier

# Banner colour (BGR) and label per tier.
_TIER_STYLE: dict[AlertTier, tuple[tuple[int, int, int], str]] = {
    AlertTier.NONE: ((0, 150, 0), "ALERT"),
    AlertTier.VISUAL: ((0, 200, 200), "LOW VIGILANCE"),
    AlertTier.AUDIBLE: ((0, 140, 255), "DROWSY"),
    AlertTier.ALARM: ((0, 0, 255), "DROWSY: PULL OVER WHEN SAFE"),
}


def draw_hud(
    frame: np.ndarray,
    tier: AlertTier,
    ear: float | None = None,
    fatigue_level: int | None = None,
    distraction_activity: str | None = None,
    distraction_tier: AlertTier | None = None,
    gaze_zone: str | None = None,
    gaze_tier: AlertTier | None = None,
) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    color, label = _TIER_STYLE[tier]

    banner_height = max(18, height // 12)
    cv2.rectangle(out, (0, 0), (width, banner_height), color, thickness=-1)
    cv2.putText(
        out, label, (8, banner_height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
    )

    if distraction_activity is not None:
        distracted = distraction_tier is not None and distraction_tier > AlertTier.NONE
        text = (
            f"DISTRACTED: {distraction_activity}"
            if distracted
            else f"attention: {distraction_activity}"
        )
        line_color = (0, 140, 255) if distracted else (200, 200, 200)
        cv2.putText(
            out,
            text,
            (8, height - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            line_color,
            1,
            cv2.LINE_AA,
        )

    if gaze_zone is not None:
        off_road = gaze_tier is not None and gaze_tier > AlertTier.NONE
        text = f"EYES OFF ROAD: {gaze_zone}" if off_road else f"gaze: {gaze_zone}"
        line_color = (0, 140, 255) if off_road else (200, 200, 200)
        cv2.putText(
            out,
            text,
            (8, height - 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            line_color,
            1,
            cv2.LINE_AA,
        )

    readouts = []
    if ear is not None:
        readouts.append(f"EAR {ear:.2f}")
    if fatigue_level is not None:
        readouts.append(f"level {fatigue_level}")
    if readouts:
        cv2.putText(
            out,
            "  ".join(readouts),
            (8, height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out
