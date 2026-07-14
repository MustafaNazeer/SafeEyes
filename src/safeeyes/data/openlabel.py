"""Parsing for OpenLABEL annotation documents as shipped with the DMD bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ActionInterval:
    action: str
    start_frame: int
    end_frame: int


def parse_openlabel_actions(source: dict[str, Any] | str | Path) -> list[ActionInterval]:
    if isinstance(source, (str, Path)):
        with open(source, encoding="utf-8") as f:
            document = json.load(f)
    else:
        document = source
    body = document.get("openlabel") if "openlabel" in document else document.get("vcd")
    if body is None:
        raise ValueError("not an OpenLABEL document: missing 'openlabel' or 'vcd' root key")
    intervals = [
        ActionInterval(str(action["type"]), int(fi["frame_start"]), int(fi["frame_end"]))
        for action in body.get("actions", {}).values()
        for fi in action.get("frame_intervals", [])
    ]
    return sorted(intervals, key=lambda i: (i.start_frame, i.end_frame, i.action))
