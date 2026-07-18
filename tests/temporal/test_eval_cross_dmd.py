"""Per-subject assembly for the DMD cross-dataset evaluation.

Aligns the extracted feature rows with frame-level drowsy labels via the recorded
frame indices, then windows them. The heavy lifting (label derivation, windowing,
metrics) is tested elsewhere; this pins the alignment glue.
"""

from __future__ import annotations

import numpy as np

from safeeyes.temporal.eval_cross_dmd import subject_windows


def _doc(action: str, start: int, end: int) -> dict:
    return {
        "openlabel": {
            "actions": {
                "0": {"type": action, "frame_intervals": [{"frame_start": start, "frame_end": end}]}
            }
        }
    }


def test_subject_windows_aligns_labels_via_frame_indices() -> None:
    # 200 feature rows sampled every 5th raw frame, so row k is raw frame 5k.
    features = np.zeros((200, 5), dtype=np.float32)
    frame_indices = np.arange(200, dtype=np.int64) * 5  # raw frames 0..995
    n_raw = 1000
    # a sustained closure over raw frames 900..960 (>= 0.5s at 30 fps) -> rows 180..192,
    # which lands only in the second window (rows 50..199), not the first (rows 0..149).
    doc = _doc("eyes_state/close", 900, 960)

    windows, labels = subject_windows(
        features,
        frame_indices,
        n_raw,
        doc,
        window_size=150,
        stride=50,
        positive_fraction=0.02,
        fps=30.0,
        min_closed_seconds=0.5,
    )

    assert len(windows) == len(labels)
    assert len(windows) == 2  # starts 0 and 50; start 100 would overrun 200 rows
    assert bool(labels[0]) is False  # first window is entirely before the closure
    assert bool(labels[1]) is True  # closure rows 180..192 fall in the second window
