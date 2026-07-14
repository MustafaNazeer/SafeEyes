from __future__ import annotations

import json
from pathlib import Path

import pytest

from safeeyes.data.openlabel import ActionInterval, parse_openlabel_actions


def _doc(actions: dict[str, dict]) -> dict:
    return {"openlabel": {"actions": actions}}


def test_actions_parse_to_one_interval_per_frame_interval() -> None:
    doc = _doc(
        {
            "0": {
                "type": "driver_actions/drinking",
                "frame_intervals": [{"frame_start": 10, "frame_end": 40}],
            },
            "1": {
                "type": "driver_actions/safe_drive",
                "frame_intervals": [
                    {"frame_start": 0, "frame_end": 9},
                    {"frame_start": 41, "frame_end": 100},
                ],
            },
        }
    )
    intervals = parse_openlabel_actions(doc)
    assert intervals == [
        ActionInterval("driver_actions/safe_drive", 0, 9),
        ActionInterval("driver_actions/drinking", 10, 40),
        ActionInterval("driver_actions/safe_drive", 41, 100),
    ]


def test_intervals_are_sorted_by_start_frame() -> None:
    doc = _doc(
        {
            "0": {
                "type": "b",
                "frame_intervals": [{"frame_start": 50, "frame_end": 60}],
            },
            "1": {
                "type": "a",
                "frame_intervals": [{"frame_start": 0, "frame_end": 10}],
            },
        }
    )
    starts = [i.start_frame for i in parse_openlabel_actions(doc)]
    assert starts == sorted(starts)


def test_vcd_root_key_is_accepted() -> None:
    doc = {
        "vcd": {
            "actions": {
                "0": {"type": "x", "frame_intervals": [{"frame_start": 1, "frame_end": 2}]}
            }
        }
    }
    assert parse_openlabel_actions(doc) == [ActionInterval("x", 1, 2)]


def test_document_without_actions_yields_empty_list() -> None:
    assert parse_openlabel_actions({"openlabel": {}}) == []


def test_unrecognized_document_root_raises() -> None:
    with pytest.raises(ValueError, match="OpenLABEL"):
        parse_openlabel_actions({"something_else": {}})


def test_path_source_is_loaded_from_disk(tmp_path: Path) -> None:
    doc = _doc({"0": {"type": "x", "frame_intervals": [{"frame_start": 3, "frame_end": 7}]}})
    p = tmp_path / "ann.json"
    p.write_text(json.dumps(doc))
    assert parse_openlabel_actions(p) == [ActionInterval("x", 3, 7)]
