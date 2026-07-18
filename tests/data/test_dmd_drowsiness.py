"""Frame-level drowsy labels derived from DMD eye-closure annotations.

The DMD drowsiness annotations mark eyes_state/close intervals separately from
blinks. A frame counts as drowsy only when it lies inside a sustained closure (a
microsleep proxy), so ordinary blinks and brief closures do not, and yawning is
excluded by design. These tests pin that criterion and the clamping of intervals
that overrun the video length.
"""

from __future__ import annotations

import numpy as np

from safeeyes.data.dmd_drowsiness import frame_drowsy_labels


def _doc(*actions: tuple[str, int, int]) -> dict:
    return {
        "openlabel": {
            "actions": {
                str(i): {
                    "type": t,
                    "frame_intervals": [{"frame_start": s, "frame_end": e}],
                }
                for i, (t, s, e) in enumerate(actions)
            }
        }
    }


def test_sustained_closure_is_drowsy() -> None:
    # 20-frame closure at 30 fps is 0.67s, above the 0.5s (15 frame) threshold.
    doc = _doc(("eyes_state/close", 10, 29))
    labels = frame_drowsy_labels(doc, n_frames=60, fps=30.0, min_closed_seconds=0.5)
    assert labels.dtype == np.bool_
    assert labels.shape == (60,)
    assert labels[10:30].all()
    assert not labels[:10].any()
    assert not labels[30:].any()


def test_short_closure_is_not_drowsy() -> None:
    # 6-frame closure at 30 fps is 0.2s, a blink-length closure, below threshold.
    doc = _doc(("eyes_state/close", 10, 15))
    labels = frame_drowsy_labels(doc, n_frames=40, fps=30.0, min_closed_seconds=0.5)
    assert not labels.any()


def test_blinks_and_yawning_are_never_drowsy() -> None:
    doc = _doc(
        ("blinks/blinking", 0, 40),
        ("yawning/Yawning with hand", 0, 40),
        ("eyes_state/open", 0, 40),
    )
    labels = frame_drowsy_labels(doc, n_frames=41, fps=30.0, min_closed_seconds=0.5)
    assert not labels.any()


def test_intervals_are_clamped_to_video_length() -> None:
    # Annotation overruns the decoded video (a real DMD quirk); must not crash.
    doc = _doc(("eyes_state/close", 30, 80))
    labels = frame_drowsy_labels(doc, n_frames=40, fps=30.0, min_closed_seconds=0.5)
    assert labels[30:40].all()
    assert labels.shape == (40,)


def test_threshold_is_configurable() -> None:
    doc = _doc(("eyes_state/close", 0, 19))  # 20 frames = 0.67s at 30 fps
    strict = frame_drowsy_labels(doc, n_frames=30, fps=30.0, min_closed_seconds=1.0)
    lenient = frame_drowsy_labels(doc, n_frames=30, fps=30.0, min_closed_seconds=0.5)
    assert not strict.any()  # 20 frames < 30-frame (1.0s) threshold
    assert lenient[0:20].all()
