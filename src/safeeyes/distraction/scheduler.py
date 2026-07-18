"""Nth-frame scheduling and EMA smoothing for the distraction classifier.

The distraction backbone is far heavier than the per frame geometric path, so it
runs on a cadence (every Nth frame) rather than every frame, protecting the fps
budget. Its class probabilities are smoothed with an exponential moving average
so a single noisy frame does not flip the alert, and between runs the last
smoothed state is held. The scheduler is torch free: it takes any classifier
callable from a preprocessed frame to a probability vector, so the ONNX adapter
drops in on the Pi with no PyTorch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Classifier = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DistractionState:
    probabilities: np.ndarray
    activity: str
    distracted: bool
    ran: bool


class DistractionScheduler:
    def __init__(
        self,
        classifier: Classifier,
        labels: Sequence[str],
        *,
        every_n: int = 5,
        ema_alpha: float = 0.5,
        safe_label: str = "safe_drive",
    ) -> None:
        if every_n < 1:
            raise ValueError("every_n must be at least 1")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        labels = tuple(labels)
        if safe_label not in labels:
            raise ValueError(f"safe_label {safe_label!r} is not among the labels")
        self._classifier = classifier
        self._labels = labels
        self._every_n = every_n
        self._alpha = float(ema_alpha)
        self._safe_label = safe_label
        self._safe_index = labels.index(safe_label)
        self.reset()

    def reset(self) -> None:
        self._counter = 0
        self._probs: np.ndarray | None = None

    def update(self, frame: np.ndarray | None) -> DistractionState:
        self._counter += 1
        ran = False
        if frame is not None and self._counter % self._every_n == 0:
            new = np.asarray(self._classifier(frame), dtype=np.float32)
            if self._probs is None:
                self._probs = new
            else:
                self._probs = self._alpha * new + (1.0 - self._alpha) * self._probs
            ran = True
        return self._state(ran)

    def _state(self, ran: bool) -> DistractionState:
        if self._probs is None:
            n = len(self._labels)
            uniform = np.full(n, 1.0 / n, dtype=np.float32)
            return DistractionState(uniform, self._safe_label, False, ran)
        index = int(np.argmax(self._probs))
        return DistractionState(
            self._probs.copy(),
            self._labels[index],
            index != self._safe_index,
            ran,
        )
