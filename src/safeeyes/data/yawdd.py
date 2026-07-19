"""YawDD filename parsing and manifest building.

YawDD ships no annotation files; the video category lives entirely in the file
name. Mirror camera clips are named ``<num>-<Gender><Glasses>-<Action>.avi``,
where the action token may join two activities with an ampersand (for example
``Talking&Yawning``); the archive also contains one lowercase variant
(``Talking&yawning``) that must normalize to the same two actions. Dash camera
clips drop the action token entirely (``<num>-<Gender><Glasses>.avi``), because
each Dash recording already contains driving, talking, and yawning scenes back
to back in a single video, per the dataset README. There is no per file action
to extract for Dash, so every Dash sample is labeled with a single sentinel,
``AllActivities``, rather than a specific action.

The real archive listing also carries two naming anomalies among the Dash
clips: four file names end in a doubled extension (``.avi.avi``) and one file
name has a trailing space before the extension (``13-MaleNoGlasses .avi``).
Both are accepted as is, because the manifest keeps the exact on disk file
name so extraction can find the file; nothing is renamed on disk.

Mirror and Dash are two different camera geometries and are always built as
separate manifests. A single manifest is never a mix of the two.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from safeeyes.data.manifest import write_manifest
from safeeyes.data.splits import Sample

YAWNING_KEYWORD = "Yawning"
DASH_ALL_ACTIVITIES = "AllActivities"

_MIRROR_CAMERA = "mirror"
_DASH_CAMERA = "dash"

_NAME_RE = re.compile(
    r"^(?P<num>\d+)-(?P<gender>Male|Female)"
    r"(?P<glasses>NoGlasses|SunGlasses|GlassesBeard|Glassesmoustache|Glasses)"
    r"(?:-(?P<action>[A-Za-z&]+))?\s*(?:\.avi)+$"
)


def parse_yawdd_filename(name: str) -> dict[str, object]:
    match = _NAME_RE.match(name)
    if match is None:
        raise ValueError(f"unrecognized YawDD filename: {name!r}")
    raw_action = match["action"]
    actions = [token.capitalize() for token in raw_action.split("&")] if raw_action else []
    return {
        "subject_num": int(match["num"]),
        "gender": match["gender"],
        "glasses": match["glasses"],
        "actions": actions,
    }


def is_yawning(actions: Sequence[str]) -> bool:
    return YAWNING_KEYWORD in actions


def build_yawdd_manifest(video_root: Path, out_path: Path, camera: str) -> list[Sample]:
    if camera not in (_MIRROR_CAMERA, _DASH_CAMERA):
        raise ValueError(f"unknown YawDD camera {camera!r}; expected 'mirror' or 'dash'")

    video_root = Path(video_root)
    samples: list[Sample] = []
    for path in sorted(video_root.rglob("*.avi")):
        parsed = parse_yawdd_filename(path.name)
        gender = cast(str, parsed["gender"])
        subject_num = cast(int, parsed["subject_num"])
        actions = cast(list[str], parsed["actions"])

        if camera == _MIRROR_CAMERA:
            if not actions:
                raise ValueError(f"mirror video has no action token: {path.name}")
            label = "&".join(actions)
        else:
            label = DASH_ALL_ACTIVITIES

        samples.append(
            Sample(
                sample_id=str(path.relative_to(video_root)),
                subject_id=f"{gender}{subject_num}",
                label=label,
            )
        )

    if not samples:
        raise ValueError(f"no YawDD videos found under {video_root}")

    write_manifest(samples, out_path)
    return samples
