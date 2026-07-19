import numpy as np
import pytest

from safeeyes.temporal.yawn_validation import MAR_YAWN_THRESHOLD, derive_mar_threshold


def test_derive_mar_threshold_is_the_requested_percentile():
    mar = np.arange(1000, dtype=float) / 1000.0
    assert derive_mar_threshold(mar, percentile=99.0) == pytest.approx(0.98901, abs=1e-4)


def test_derive_mar_threshold_rejects_empty():
    with pytest.raises(ValueError):
        derive_mar_threshold(np.array([]))


def test_registered_threshold_is_pinned():
    assert MAR_YAWN_THRESHOLD == pytest.approx(0.616703, abs=1e-6)
