from pathlib import Path

import pytest

from safeeyes.data.splits import assert_subject_independent, subject_independent_split
from safeeyes.data.uta_rldd import build_uta_manifest, uta_label_from_stem


def _make_tree(root: Path) -> None:
    # Two folds, each with subjects holding one video per class.
    layout = {
        "Fold1/001": ["0.mp4", "5.mp4", "10.mp4"],
        "Fold1/002": ["0.mov", "5.mov", "10.mov"],
        "Fold2/003": ["0.mp4", "5.mp4", "10.mp4"],
    }
    for rel, files in layout.items():
        d = root / rel
        d.mkdir(parents=True)
        for f in files:
            (d / f).write_bytes(b"")
    # Noise that must be ignored.
    (root / "Fold1/001/notes.txt").write_text("ignore me")
    (root / "README.md").write_text("ignore me")


def test_label_mapping_matches_uta_convention() -> None:
    assert uta_label_from_stem("0") == "alert"
    assert uta_label_from_stem("5") == "low_vigilance"
    assert uta_label_from_stem("10") == "drowsy"


def test_unknown_label_stem_raises() -> None:
    with pytest.raises(ValueError):
        uta_label_from_stem("7")


def test_builds_one_sample_per_class_video(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    manifest = build_uta_manifest(tmp_path)
    assert len(manifest) == 9


def test_subject_id_comes_from_parent_directory(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    manifest = build_uta_manifest(tmp_path)
    assert {s.subject_id for s in manifest} == {"001", "002", "003"}


def test_labels_are_mapped_and_non_video_files_ignored(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    manifest = build_uta_manifest(tmp_path)
    labels = sorted(s.label for s in manifest)
    assert labels == sorted(["alert", "low_vigilance", "drowsy"] * 3)


def test_manifest_splits_without_subject_leakage(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    manifest = build_uta_manifest(tmp_path)
    split = subject_independent_split(manifest, ratios=(0.66, 0.0, 0.34), seed=0)
    assert_subject_independent(split)


def test_empty_root_yields_empty_manifest(tmp_path: Path) -> None:
    assert build_uta_manifest(tmp_path) == []
