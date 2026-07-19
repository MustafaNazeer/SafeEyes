from pathlib import Path

import pytest

from safeeyes.data.manifest import read_manifest
from safeeyes.data.yawdd import (
    DASH_ALL_ACTIVITIES,
    build_yawdd_manifest,
    is_yawning,
    parse_yawdd_filename,
)


def test_parse_plain_mirror_filename() -> None:
    parsed = parse_yawdd_filename("1-MaleNoGlasses-Yawning.avi")
    assert parsed == {
        "subject_num": 1,
        "gender": "Male",
        "glasses": "NoGlasses",
        "actions": ["Yawning"],
    }


def test_parse_ampersand_filename_yields_both_actions() -> None:
    parsed = parse_yawdd_filename("30-FemaleGlasses-Talking&Yawning.avi")
    assert parsed["actions"] == ["Talking", "Yawning"]


def test_parse_lowercase_ampersand_action_is_normalized() -> None:
    parsed = parse_yawdd_filename("12-MaleNoGlasses-Talking&yawning.avi")
    assert parsed["actions"] == ["Talking", "Yawning"]


def test_parse_normal_action() -> None:
    parsed = parse_yawdd_filename("5-FemaleSunGlasses-Normal.avi")
    assert parsed["actions"] == ["Normal"]


@pytest.mark.parametrize(
    ("name", "expected_glasses"),
    [
        ("2-MaleGlassesBeard-Talking.avi", "GlassesBeard"),
        ("3-MaleGlassesmoustache-Talking.avi", "Glassesmoustache"),
        ("4-FemaleSunGlasses-Talking.avi", "SunGlasses"),
        ("6-MaleGlasses-Talking.avi", "Glasses"),
        ("7-MaleNoGlasses-Talking.avi", "NoGlasses"),
    ],
)
def test_glasses_alternation_does_not_shadow_longer_variants(
    name: str, expected_glasses: str
) -> None:
    parsed = parse_yawdd_filename(name)
    assert parsed["glasses"] == expected_glasses


def test_parse_dash_filename_has_no_action_and_empty_actions() -> None:
    parsed = parse_yawdd_filename("8-MaleNoGlasses.avi")
    assert parsed == {
        "subject_num": 8,
        "gender": "Male",
        "glasses": "NoGlasses",
        "actions": [],
    }


def test_parse_accepts_double_avi_extension_anomaly() -> None:
    parsed = parse_yawdd_filename("9-FemaleGlasses.avi.avi")
    assert parsed["subject_num"] == 9
    assert parsed["actions"] == []


def test_parse_accepts_trailing_space_anomaly() -> None:
    parsed = parse_yawdd_filename("13-MaleNoGlasses .avi")
    assert parsed["subject_num"] == 13
    assert parsed["gender"] == "Male"
    assert parsed["glasses"] == "NoGlasses"
    assert parsed["actions"] == []


def test_parse_rejects_unrecognized_name() -> None:
    with pytest.raises(ValueError):
        parse_yawdd_filename("notes.txt")


def test_is_yawning() -> None:
    assert is_yawning(["Yawning"]) is True
    assert is_yawning(["Talking", "Yawning"]) is True
    assert is_yawning(["Normal"]) is False
    assert is_yawning(["Talking"]) is False


def _make_mirror_tree(root: Path) -> None:
    male = root / "Mirror" / "Male"
    female = root / "Mirror" / "Female"
    male.mkdir(parents=True)
    female.mkdir(parents=True)
    (male / "1-MaleNoGlasses-Normal.avi").write_bytes(b"")
    (male / "1-MaleNoGlasses-Yawning.avi").write_bytes(b"")
    (female / "30-FemaleGlasses-Talking&Yawning.avi").write_bytes(b"")
    (root / "README.md").write_text("ignore me")


def _make_dash_tree(root: Path) -> None:
    male = root / "Dash" / "Male"
    female = root / "Dash" / "Female"
    male.mkdir(parents=True)
    female.mkdir(parents=True)
    (male / "8-MaleNoGlasses.avi").write_bytes(b"")
    (male / "9-MaleGlasses.avi.avi").write_bytes(b"")
    (female / "13-FemaleNoGlasses .avi").write_bytes(b"")


def test_build_mirror_manifest_writes_expected_rows(tmp_path: Path) -> None:
    _make_mirror_tree(tmp_path)
    out_path = tmp_path / "out" / "mirror.csv"
    samples = build_yawdd_manifest(tmp_path / "Mirror", out_path, camera="mirror")
    assert len(samples) == 3

    written = read_manifest(out_path)
    assert len(written) == 3
    labels = sorted(s.label for s in written)
    assert labels == ["Normal", "Talking&Yawning", "Yawning"]
    subject_ids = {s.subject_id for s in written}
    assert subject_ids == {"Male1", "Female30"}
    sample_ids = {s.sample_id for s in written}
    assert sample_ids == {
        "Male/1-MaleNoGlasses-Normal.avi",
        "Male/1-MaleNoGlasses-Yawning.avi",
        "Female/30-FemaleGlasses-Talking&Yawning.avi",
    }


def test_build_dash_manifest_uses_all_activities_label_and_keeps_anomalous_names(
    tmp_path: Path,
) -> None:
    _make_dash_tree(tmp_path)
    out_path = tmp_path / "out" / "dash.csv"
    samples = build_yawdd_manifest(tmp_path / "Dash", out_path, camera="dash")
    assert len(samples) == 3
    assert all(s.label == DASH_ALL_ACTIVITIES for s in samples)

    written = read_manifest(out_path)
    sample_ids = {s.sample_id for s in written}
    assert sample_ids == {
        "Male/8-MaleNoGlasses.avi",
        "Male/9-MaleGlasses.avi.avi",
        "Female/13-FemaleNoGlasses .avi",
    }


def test_build_manifest_raises_on_mirror_video_with_no_action(tmp_path: Path) -> None:
    mirror = tmp_path / "Mirror"
    mirror.mkdir()
    (mirror / "1-MaleNoGlasses.avi").write_bytes(b"")
    with pytest.raises(ValueError):
        build_yawdd_manifest(mirror, tmp_path / "out.csv", camera="mirror")


def test_build_manifest_raises_on_unknown_camera(tmp_path: Path) -> None:
    _make_mirror_tree(tmp_path)
    with pytest.raises(ValueError):
        build_yawdd_manifest(tmp_path / "Mirror", tmp_path / "out.csv", camera="front")


def test_build_manifest_raises_on_empty_tree(tmp_path: Path) -> None:
    empty = tmp_path / "Empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        build_yawdd_manifest(empty, tmp_path / "out.csv", camera="mirror")
