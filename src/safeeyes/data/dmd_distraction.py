"""Manifest builder for the DMD distraction bundle (body camera, driver actions)."""

from __future__ import annotations

from pathlib import Path

from safeeyes.data.intervals import IntervalSample
from safeeyes.data.openlabel import parse_openlabel_actions

_ANNOTATION_SUFFIX = "_rgb_ann_distraction.json"
_BODY_SUFFIX = "_rgb_body.mp4"
_PREFIX = "driver_actions/"

DMD_DISTRACTION_ACTIONS = frozenset(
    {
        "driver_actions/change_gear",
        "driver_actions/drinking",
        "driver_actions/hair_and_makeup",
        "driver_actions/phonecall_left",
        "driver_actions/phonecall_right",
        "driver_actions/radio",
        "driver_actions/reach_backseat",
        "driver_actions/reach_side",
        "driver_actions/safe_drive",
        "driver_actions/standstill_or_waiting",
        "driver_actions/talking_to_passenger",
        "driver_actions/texting_left",
        "driver_actions/texting_right",
    }
)
SKIPPED_ACTIONS = frozenset({"driver_actions/unclassified"})


def build_dmd_distraction_manifest(
    root: str | Path,
    allowed_actions: frozenset[str] = DMD_DISTRACTION_ACTIONS,
) -> list[IntervalSample]:
    root = Path(root)
    samples: list[IntervalSample] = []
    for annotation_path in sorted(root.rglob(f"*{_ANNOTATION_SUFFIX}")):
        video_path = annotation_path.with_name(
            annotation_path.name.replace(_ANNOTATION_SUFFIX, _BODY_SUFFIX)
        )
        if not video_path.is_file():
            raise FileNotFoundError(f"body video missing for annotations: {video_path}")
        subject_dir = annotation_path.parent.parent
        subject_id = f"{subject_dir.parent.name}_{subject_dir.name}"
        video_rel = video_path.relative_to(root)
        for interval in parse_openlabel_actions(annotation_path):
            if not interval.action.startswith(_PREFIX) or interval.action in SKIPPED_ACTIONS:
                continue
            if interval.action not in allowed_actions:
                raise ValueError(
                    f"unknown driver action type {interval.action!r} in {annotation_path}"
                )
            samples.append(
                IntervalSample(
                    sample_id=f"{video_rel}#{interval.start_frame}-{interval.end_frame}",
                    subject_id=subject_id,
                    label=interval.action.removeprefix(_PREFIX),
                    start_frame=interval.start_frame,
                    end_frame=interval.end_frame,
                )
            )
    return samples
