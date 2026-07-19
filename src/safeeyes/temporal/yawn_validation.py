"""Video level validation of the geometric MAR yawn signal.

The detection threshold is pre-registered: derived from UTA train subject MAR
statistics only (the 99th percentile of all per frame MAR values across the
train split features), fixed before any YawDD data was listed, extracted, or
scored, and never revised after. YawDD is therefore a purely held out test set.
"""

from __future__ import annotations

import numpy as np


def derive_mar_threshold(mar_values: np.ndarray, percentile: float = 99.0) -> float:
    values = np.asarray(mar_values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot derive a threshold from no MAR values")
    return float(np.percentile(values, percentile))


MAR_YAWN_THRESHOLD: float = 0.616703
