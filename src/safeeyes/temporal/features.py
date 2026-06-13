"""Window level feature aggregation.

Turns the per frame signals collected over a window into the interpretable
fatigue features the temporal stage uses: the proportion of eye closure, blink
and yawn and nod event counts and rates, and mean signal levels. An event is an
onset: a transition into the state, so a sustained closure counts once, not once
per frame. These definitions are deliberately simple and tested against known
sequences.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from safeeyes.perception.geometry import perclos


def _in_state(series: Sequence[float], threshold: float, mode: str) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if mode == "below":
        return arr <= threshold
    if mode == "above":
        return arr >= threshold
    raise ValueError(f"mode must be 'below' or 'above', got {mode!r}")


def count_onsets(series: Sequence[float], threshold: float, mode: str) -> int:
    state = _in_state(series, threshold, mode)
    if state.size == 0:
        return 0
    previous = np.concatenate(([False], state[:-1]))
    onsets = state & ~previous
    return int(np.count_nonzero(onsets))


def mean_closure_duration(series: Sequence[float], threshold: float) -> float:
    state = _in_state(series, threshold, "below")
    runs: list[int] = []
    current = 0
    for closed in state:
        if closed:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return float(np.mean(runs)) if runs else 0.0


def event_rate_per_minute(count: int, n_frames: int, fps: float) -> float:
    if n_frames <= 0 or fps <= 0:
        return 0.0
    seconds = n_frames / fps
    return count / seconds * 60.0


def summarize_window(
    ear: Sequence[float],
    mar: Sequence[float],
    pitch: Sequence[float],
    fps: float,
    ear_threshold: float,
    mar_threshold: float,
    nod_threshold: float,
) -> dict[str, float]:
    n = len(ear)
    abs_pitch = np.abs(np.asarray(pitch, dtype=float))
    blink_count = count_onsets(ear, ear_threshold, "below")
    yawn_count = count_onsets(mar, mar_threshold, "above")
    nod_count = count_onsets(abs_pitch, nod_threshold, "above")
    return {
        "perclos": perclos(ear, ear_threshold),
        "blink_count": blink_count,
        "yawn_count": yawn_count,
        "nod_count": nod_count,
        "blink_rate_per_min": event_rate_per_minute(blink_count, n, fps),
        "yawn_rate_per_min": event_rate_per_minute(yawn_count, n, fps),
        "nod_rate_per_min": event_rate_per_minute(nod_count, n, fps),
        "mean_blink_duration": mean_closure_duration(ear, ear_threshold),
        "mean_ear": float(np.mean(ear)) if n else 0.0,
        "mean_mar": float(np.mean(mar)) if len(mar) else 0.0,
        "mean_abs_pitch": float(np.mean(abs_pitch)) if abs_pitch.size else 0.0,
    }
