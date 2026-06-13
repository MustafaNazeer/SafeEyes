import numpy as np

from safeeyes.temporal.window import assemble_windowed_dataset, make_windows


def test_make_windows_overlapping() -> None:
    seq = np.arange(10 * 2).reshape(10, 2).astype(float)  # T=10, F=2
    windows, labels = make_windows(seq, label=1, window_size=4, stride=2)
    # starts 0, 2, 4, 6 produce full windows; start 8 would overrun
    assert len(windows) == 4
    assert all(w.shape == (4, 2) for w in windows)
    assert labels == [1, 1, 1, 1]


def test_make_windows_first_window_content() -> None:
    seq = np.arange(10 * 1).reshape(10, 1).astype(float)
    windows, _ = make_windows(seq, label=0, window_size=3, stride=3)
    assert windows[0].flatten().tolist() == [0.0, 1.0, 2.0]
    assert windows[1].flatten().tolist() == [3.0, 4.0, 5.0]


def test_make_windows_too_short_yields_nothing() -> None:
    seq = np.zeros((2, 3))
    windows, labels = make_windows(seq, label=2, window_size=5, stride=1)
    assert windows == [] and labels == []


def test_assemble_windowed_dataset_stacks_sequences() -> None:
    seq_a = np.zeros((8, 2))  # window 4 stride 4 -> 2 windows
    seq_b = np.ones((4, 2))  # -> 1 window
    x, y = assemble_windowed_dataset(
        [(seq_a, 0), (seq_b, 1)], window_size=4, stride=4
    )
    assert x.shape == (3, 4, 2)
    assert y.tolist() == [0, 0, 1]


def test_assemble_windowed_dataset_empty_when_all_too_short() -> None:
    x, y = assemble_windowed_dataset([(np.zeros((2, 2)), 0)], window_size=5, stride=1)
    assert x.shape == (0, 5, 2)
    assert y.shape == (0,)
