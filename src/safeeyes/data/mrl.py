"""MRL Eye Dataset filename parsing.

Each image filename encodes its annotations as underscore separated fields, in
the order documented by the dataset authors:

    subjectID_imageID_gender_glasses_eyeState_reflections_lighting_sensorID.ext

Eye state is the fifth field: 0 means closed, 1 means open. The subject ID
(first field) is what the subject independent split keys on, so an eye state
classifier never sees the same subject in both training and evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from safeeyes.data.splits import Sample

_FIELD_COUNT = 8
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class MrlRecord:
    path: Path
    subject_id: str
    image_id: str
    gender: int
    glasses: int
    eye_state: int
    reflections: int
    lighting: int
    sensor_id: str

    @property
    def is_open(self) -> bool:
        return self.eye_state == 1


def parse_mrl_filename(filename: str | Path) -> MrlRecord:
    path = Path(filename)
    fields = path.stem.split("_")
    if len(fields) != _FIELD_COUNT:
        raise ValueError(
            f"expected {_FIELD_COUNT} underscore separated fields in MRL filename, "
            f"got {len(fields)}: {path.name!r}"
        )
    subject_id, image_id, gender, glasses, eye_state, reflections, lighting, sensor_id = fields
    return MrlRecord(
        path=path,
        subject_id=subject_id,
        image_id=image_id,
        gender=int(gender),
        glasses=int(glasses),
        eye_state=int(eye_state),
        reflections=int(reflections),
        lighting=int(lighting),
        sensor_id=sensor_id,
    )


def to_sample(record: MrlRecord) -> Sample:
    return Sample(
        sample_id=record.path.name,
        subject_id=record.subject_id,
        label="open" if record.is_open else "closed",
    )


def build_mrl_manifest(
    root: str | Path,
    image_extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> list[Sample]:
    root = Path(root)
    exts = {e.lower() for e in image_extensions}
    samples: list[Sample] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        samples.append(to_sample(parse_mrl_filename(path)))
    return samples
