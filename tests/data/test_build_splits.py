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
