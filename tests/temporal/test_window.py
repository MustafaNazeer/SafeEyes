import numpy as np
import pytest

from safeeyes.temporal.window import FeatureWindow


def test_new_window_is_empty_and_not_full() -> None:
    w = FeatureWindow(capacity=3, n_features=2)
    assert len(w) == 0
    assert w.is_full is False


def test_push_accumulates_until_full() -> None:
    w = FeatureWindow(capacity=3, n_features=2)
    w.push([1.0, 2.0])
    assert len(w) == 1 and w.is_full is False
    w.push([3.0, 4.0])
    w.push([5.0, 6.0])
    assert len(w) == 3 and w.is_full is True


def test_capacity_is_never_exceeded_and_oldest_drops_first() -> None:
    w = FeatureWindow(capacity=2, n_features=1)
    w.push([1.0])
    w.push([2.0])
    w.push([3.0])  # evicts [1.0]
    assert len(w) == 2
    assert w.as_array().flatten().tolist() == [2.0, 3.0]


def test_as_array_shape_and_order() -> None:
    w = FeatureWindow(capacity=3, n_features=2)
    w.push([1.0, 2.0])
    w.push([3.0, 4.0])
    arr = w.as_array()
    assert arr.shape == (2, 2)
    assert arr.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_push_wrong_feature_count_raises() -> None:
    w = FeatureWindow(capacity=3, n_features=2)
    with pytest.raises(ValueError):
        w.push([1.0])


def test_clear_empties_the_window() -> None:
    w = FeatureWindow(capacity=2, n_features=1)
    w.push([1.0])
    w.clear()
    assert len(w) == 0
    assert w.as_array().shape == (0, 1)


def test_as_array_on_empty_window_has_zero_rows() -> None:
    w = FeatureWindow(capacity=2, n_features=3)
    assert w.as_array().shape == (0, 3)
    assert isinstance(w.as_array(), np.ndarray)
