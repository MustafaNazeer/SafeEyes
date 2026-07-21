import re
import subprocess
import tarfile

import pytest

from safeeyes.data.gaze_archives import (
    ARCHIVE_GLOB,
    WANTED_MEMBERS,
    extract_archive,
    find_archives,
    main,
)


def _make_archive(path, subject: str) -> None:
    root = path.parent / f"src-{subject}"
    session = root / "dmd" / subject[0:2] / subject.split("-")[1] / "s6"
    session.mkdir(parents=True)
    stamp = "2019-04-09T10;35;56+02;00"
    for name in (
        f"{subject}_s6_{stamp}_rgb_face.mp4",
        f"{subject}_s6_{stamp}_rgb_ann_gaze.json",
        f"{subject}_s6_{stamp}_rgb_body.mp4",
        f"{subject}_s6_{stamp}_rgb_ann_hands.json",
    ):
        (session / name).write_text("x")
    with tarfile.open(path, "w:gz") as tar:
        tar.add(root / "dmd", arcname="dmd")


def test_only_the_face_video_and_gaze_annotation_are_extracted(tmp_path) -> None:
    archive = tmp_path / "dmd-dataset-gaze-gZ-36.tar.gz"
    _make_archive(archive, "gZ-36")
    out = tmp_path / "out"
    out.mkdir()

    extract_archive(archive, out)

    names = sorted(p.name for p in out.rglob("*") if p.is_file())
    assert len(names) == 2
    assert any(n.endswith("_rgb_face.mp4") for n in names)
    assert any(n.endswith("_rgb_ann_gaze.json") for n in names)
    assert not any(n.endswith("_rgb_body.mp4") for n in names)
    assert not any(n.endswith("_rgb_ann_hands.json") for n in names)


def test_wanted_members_are_pinned() -> None:
    assert WANTED_MEMBERS == ("*_rgb_face.mp4", "*_rgb_ann_gaze.json")


def test_missing_archives_raise_rather_than_silently_doing_nothing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=re.escape(ARCHIVE_GLOB)):
        find_archives(tmp_path)


def test_archives_are_returned_in_a_stable_order(tmp_path) -> None:
    for subject in ("gZ-36", "gA-1", "gC-13"):
        _make_archive(tmp_path / f"dmd-dataset-gaze-{subject}.tar.gz", subject)
    found = [p.name for p in find_archives(tmp_path)]
    assert found == sorted(found)


def test_skip_existing_does_not_re_extract(tmp_path, capsys) -> None:
    archive = tmp_path / "dmd-dataset-gaze-gZ-36.tar.gz"
    _make_archive(archive, "gZ-36")
    out = tmp_path / "out"

    assert main(["--archive-dir", str(tmp_path), "--out-dir", str(out)]) == 0
    assert main(["--archive-dir", str(tmp_path), "--out-dir", str(out), "--skip-existing"]) == 0

    captured = capsys.readouterr().out
    assert "already extracted, skipping" in captured
    assert "extracted 0 of 1 archives" in captured


def test_a_failing_tar_call_is_not_swallowed(tmp_path) -> None:
    archive = tmp_path / "dmd-dataset-gaze-gZ-36.tar.gz"
    archive.write_text("not a tarball")
    with pytest.raises(subprocess.CalledProcessError):
        extract_archive(archive, tmp_path)
