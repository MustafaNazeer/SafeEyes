from pathlib import Path

import pytest

from safeeyes.data.build_splits import build_dataset_splits, main
from safeeyes.data.splits import assert_subject_independent


def _uta_tree(root: Path) -> None:
    for subject in ("001", "002", "003", "004"):
        d = root / "Fold1" / subject
        d.mkdir(parents=True)
        for stem in ("0", "5", "10"):
            (d / f"{stem}.mp4").write_bytes(b"")


def _mrl_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for subj in range(1, 5):
        for img in range(3):
            state = img % 2
            (root / f"s{subj:04d}_{img:05d}_0_0_{state}_0_1_01.png").write_bytes(b"")


def test_uta_end_to_end_writes_subject_independent_split(tmp_path: Path) -> None:
    root, out = tmp_path / "raw", tmp_path / "splits"
    _uta_tree(root)
    split = build_dataset_splits("uta-rldd", root, out, ratios=(0.5, 0.0, 0.5), seed=0)
    assert_subject_independent(split)
    assert (out / "train.csv").exists() and (out / "test.csv").exists()


def test_mrl_end_to_end(tmp_path: Path) -> None:
    root, out = tmp_path / "raw", tmp_path / "splits"
    _mrl_tree(root)
    split = build_dataset_splits("mrl", root, out, ratios=(0.5, 0.0, 0.5), seed=0)
    assert_subject_independent(split)
    assert len(split.train) + len(split.test) == 12


def test_unknown_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_dataset_splits("not-a-dataset", tmp_path, tmp_path, seed=0)


def test_empty_root_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no samples"):
        build_dataset_splits("uta-rldd", tmp_path, tmp_path, seed=0)


def test_cli_main_builds_split(tmp_path: Path) -> None:
    root, out = tmp_path / "raw", tmp_path / "splits"
    _uta_tree(root)
    exit_code = main(
        ["--dataset", "uta-rldd", "--root", str(root), "--out", str(out), "--seed", "0"]
    )
    assert exit_code == 0
    assert (out / "summary.json").exists()


def test_dmd_distraction_splits_end_to_end(tmp_path: Path) -> None:
    import json as _json

    root = tmp_path / "dmd"
    for group, subject in [("gA", "1"), ("gA", "5"), ("gB", "6"), ("gB", "7"), ("gC", "13")]:
        d = root / group / subject / "s1"
        d.mkdir(parents=True)
        stem = f"{group}_{subject}_s1_2019-01-01T00;00;00+01;00_rgb"
        (d / f"{stem}_ann_distraction.json").write_text(
            _json.dumps(
                {
                    "openlabel": {
                        "actions": {
                            "0": {
                                "type": "driver_actions/drinking",
                                "frame_intervals": [{"frame_start": 0, "frame_end": 50}],
                            },
                            "1": {
                                "type": "driver_actions/safe_drive",
                                "frame_intervals": [{"frame_start": 51, "frame_end": 200}],
                            },
                        }
                    }
                }
            )
        )
        (d / f"{stem}_body.mp4").write_bytes(b"")
    out = tmp_path / "splits"
    exit_code = main(
        [
            "--dataset", "dmd-distraction",
            "--root", str(root),
            "--out", str(out),
            "--ratios", "0.7", "0.0", "0.3",
        ]
    )
    assert exit_code == 0
    train_header = (out / "train.csv").read_text().splitlines()[0]
    assert train_header == "sample_id,subject_id,label,start_frame,end_frame"
    from safeeyes.data.intervals import read_interval_manifest

    train_subjects = {s.subject_id for s in read_interval_manifest(out / "train.csv")}
    test_subjects = {s.subject_id for s in read_interval_manifest(out / "test.csv")}
    assert train_subjects and test_subjects
    assert train_subjects.isdisjoint(test_subjects)
