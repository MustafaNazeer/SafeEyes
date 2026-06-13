from pathlib import Path

import pytest

from safeeyes.data.mrl import (
    build_mrl_manifest,
    parse_mrl_filename,
    to_sample,
)


def test_parses_all_fields_from_open_eye_filename() -> None:
    record = parse_mrl_filename("s0001_00123_1_0_1_2_1_03.png")
    assert record.subject_id == "s0001"
    assert record.image_id == "00123"
    assert record.gender == 1
    assert record.glasses == 0
    assert record.eye_state == 1
    assert record.reflections == 2
    assert record.lighting == 1
    assert record.sensor_id == "03"


def test_is_open_true_when_eye_state_is_one() -> None:
    assert parse_mrl_filename("s0001_00123_1_0_1_2_1_03.png").is_open is True


def test_is_open_false_when_eye_state_is_zero() -> None:
    assert parse_mrl_filename("s0002_00001_0_0_0_0_0_01.png").is_open is False


def test_accepts_a_full_path_and_keeps_it() -> None:
    record = parse_mrl_filename("/data/mrl/s0005_00009_0_1_0_1_0_02.png")
    assert record.subject_id == "s0005"
    assert record.path == Path("/data/mrl/s0005_00009_0_1_0_1_0_02.png")


def test_raises_on_wrong_field_count() -> None:
    with pytest.raises(ValueError):
        parse_mrl_filename("s0001_00001_0_0.png")


def test_to_sample_maps_subject_and_open_closed_label() -> None:
    record = parse_mrl_filename("s0007_00042_1_1_1_0_1_01.png")
    sample = to_sample(record)
    assert sample.subject_id == "s0007"
    assert sample.label == "open"
    assert sample.sample_id == "s0007_00042_1_1_1_0_1_01.png"


def test_to_sample_uses_closed_label_for_closed_eye() -> None:
    record = parse_mrl_filename("s0007_00043_1_1_0_0_1_01.png")
    assert to_sample(record).label == "closed"


def test_build_mrl_manifest_walks_images_and_ignores_other_files(tmp_path: Path) -> None:
    (tmp_path / "s0001_00001_0_0_1_0_1_01.png").write_bytes(b"")
    (tmp_path / "s0001_00002_0_0_0_0_1_01.png").write_bytes(b"")
    (tmp_path / "s0002_00001_1_0_1_0_1_02.png").write_bytes(b"")
    (tmp_path / "annotations.txt").write_text("ignore")
    manifest = build_mrl_manifest(tmp_path)
    assert len(manifest) == 3
    assert {s.subject_id for s in manifest} == {"s0001", "s0002"}
    assert sorted(s.label for s in manifest) == ["closed", "open", "open"]
