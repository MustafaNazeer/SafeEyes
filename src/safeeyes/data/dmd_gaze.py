"""Per frame gaze zone labels from DMD OpenLABEL annotations.

The gaze annotation carries three mutually exclusive levels (Occlusion, Gaze
zone, Blinks); only the gaze zone level is consumed here.

Frames labelled ``not_valid`` are excluded rather than mapped to a class,
because the label marks unusable data rather than a direction of gaze. That
exclusion is large: on the subjects sampled while designing this phase it
covered roughly 62 percent of every recording, so any number derived from these
labels has to report it.

Intervals are clamped to the video length, matching the DMD annotation overrun
behaviour measured across every session in the distraction work.
"""

from __future__ import annotations

GAZE_ZONES: tuple[str, ...] = (
    "left_mirror",
    "left",
    "front",
    "center_mirror",
    "front_right",
    "right_mirror",
    "right",
    "infotainment",
    "steering_wheel",
)

GAZE_PREFIX = "gaze_zone/"
EXCLUDED_ZONE = "not_valid"


def is_eyes_on_road(zone: str) -> bool:
    """Only a forward gaze counts as eyes on road.

    Mirror checks are legitimate driving glances but are deliberately not
    counted here: treating them as on road would bake a driving behaviour
    judgement into the label that the dataset does not assert. The alert track
    relies on a duration requirement rather than this predicate to avoid firing
    on normal scanning.
    """
    return zone == "front"


def frame_gaze_labels(annotation: dict, n_frames: int) -> dict[int, str]:
    root = annotation.get("openlabel", annotation)
    labels: dict[int, str] = {}
    for action in root.get("actions", {}).values():
        action_type = str(action.get("type", ""))
        if not action_type.startswith(GAZE_PREFIX):
            continue
        zone = action_type[len(GAZE_PREFIX) :]
        if zone == EXCLUDED_ZONE:
            continue
        if zone not in GAZE_ZONES:
            raise ValueError(f"unknown gaze zone {zone!r}")
        for interval in action.get("frame_intervals", []):
            start = int(interval["frame_start"])
            end = min(int(interval["frame_end"]), n_frames - 1)
            for frame in range(start, end + 1):
                labels[frame] = zone
    return labels
