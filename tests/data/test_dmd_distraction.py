from __future__ import annotations

import json
from pathlib import Path

import pytest

from safeeyes.data.dmd_distraction import (
    DMD_DISTRACTION_ACTIONS,
    build_dmd_distraction_manifest,
)


def _write_session(
    root: Path,
    group: str,
    subject: str,
    session: str,
    actions: dict[str, dict],
    with_body_video: bool = True,
) -> None:
    d = root / group / subject / session
    d.mkdir(parents=True)
    stem = f"{group}_{subject}_{session}_2019-01-01T00;00;00+01;00_rgb"
    (d / f"{stem}_ann_distraction.json").write_text(
        json.dumps({"openlabel": {"actions": actions}})
    )
    if with_body_video:
        (d / f"{stem}_body.mp4").write_bytes(b"")


def _action(action_type: str, start: int, end: int) -> dict:
    return {"type": action_type, "frame_intervals": [{"frame_start": start, "frame_end": end}]}


def test_builder_emits_one_interval_sample_per_driver_action(tmp_path: Path) -> None:
    _write_session(
        tmp_path, "gA", "1", "s1",
        {"0": _action("driver_actions/drinking", 10, 40),
         "1": _action("driver_actions/safe_drive", 41, 90)},
    )
    samples = build_dmd_distraction_manifest(tmp_path)
    assert len(samples) == 2
    assert samples[0].subject_id == "gA_1"
    assert samples[0].label == "drinking"
    assert samples[0].start_frame == 10 and samples[0].end_frame == 40
    assert samples[0].sample_id.endswith("_rgb_body.mp4#10-40")


def test_non_driver_action_groups_are_ignored(tmp_path: Path) -> None:
    _write_session(
        tmp_path, "gA", "1", "s1",
        {"0": _action("gaze_on_road/looking_road", 0, 100),
         "1": _action("hands_using_wheel/both", 0, 100),
         "2": _action("driver_actions/radio", 5, 25)},
    )
    samples = build_dmd_distraction_manifest(tmp_path)
    assert [s.label for s in samples] == ["radio"]


def test_unclassified_intervals_are_skipped(tmp_path: Path) -> None:
    _write_session(
        tmp_path, "gA", "1", "s1",
        {"0": _action("driver_actions/unclassified", 0, 10),
         "1": _action("driver_actions/drinking", 11, 20)},
    )
    assert [s.label for s in build_dmd_distraction_manifest(tmp_path)] == ["drinking"]


def test_unknown_driver_action_type_raises(tmp_path: Path) -> None:
    _write_session(
        tmp_path, "gA", "1", "s1",
        {"0": _action("driver_actions/juggling", 0, 10)},
    )
    with pytest.raises(ValueError, match="juggling"):
        build_dmd_distraction_manifest(tmp_path)


def test_missing_body_video_raises(tmp_path: Path) -> None:
    _write_session(
        tmp_path, "gA", "1", "s1",
        {"0": _action("driver_actions/drinking", 0, 10)},
        with_body_video=False,
    )
    with pytest.raises(FileNotFoundError, match="_rgb_body.mp4"):
        build_dmd_distraction_manifest(tmp_path)


def test_subjects_come_from_directory_structure(tmp_path: Path) -> None:
    _write_session(tmp_path, "gA", "1", "s1", {"0": _action("driver_actions/drinking", 0, 10)})
    _write_session(tmp_path, "gB", "6", "s2", {"0": _action("driver_actions/radio", 0, 10)})
    subjects = {s.subject_id for s in build_dmd_distraction_manifest(tmp_path)}
    assert subjects == {"gA_1", "gB_6"}


def test_known_taxonomy_matches_verified_types() -> None:
    assert DMD_DISTRACTION_ACTIONS == frozenset(
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
    assert "driver_actions/unclassified" not in DMD_DISTRACTION_ACTIONS
