"""The torch-free edge labels must match the training labels exactly."""

from __future__ import annotations


def test_edge_labels_match_training_labels() -> None:
    from safeeyes.distraction.labels import DISTRACTION_LABELS as EDGE_LABELS
    from safeeyes.models.distraction_data import DISTRACTION_LABELS as TRAIN_LABELS

    assert EDGE_LABELS == TRAIN_LABELS


def test_edge_labels_import_without_torch() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch'] = None; "
            "from safeeyes.distraction.labels import DISTRACTION_LABELS; "
            "print(len(DISTRACTION_LABELS))",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "13"
