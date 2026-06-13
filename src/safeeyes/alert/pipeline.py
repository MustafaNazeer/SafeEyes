"""Runtime drowsiness pipeline core.

Ties the rolling feature window, the temporal classifier, and the alert state
machine into one per frame step. The classifier is injected as a callable from a
filled window to a fatigue level, so this logic is exercised without a camera or
a trained model. The camera capture, landmark detection, and on screen overlay
live in the runtime loop module; this is the decision core they drive.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import nn

from safeeyes.alert.state_machine import AlertStateMachine, AlertTier
from safeeyes.temporal.window import FeatureWindow

Classifier = Callable[[np.ndarray], int]


def make_gru_classifier(model: nn.Module) -> Classifier:
    """Wrap a trained sequence model as a window to fatigue level classifier."""

    def classify(window: np.ndarray) -> int:
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            return int(model(inputs).argmax(dim=1).item())

    return classify


class DrowsinessPipeline:
    def __init__(
        self,
        classifier: Classifier,
        window_capacity: int,
        n_features: int,
        escalate_steps: int = 3,
        de_escalate_steps: int = 8,
        alarm_after: int = 15,
    ) -> None:
        self._classifier = classifier
        self._window = FeatureWindow(window_capacity, n_features)
        self._state = AlertStateMachine(
            escalate_steps=escalate_steps,
            de_escalate_steps=de_escalate_steps,
            alarm_after=alarm_after,
        )
        self._level = 0

    def process(self, features: Sequence[float] | np.ndarray) -> AlertTier:
        self._window.push(features)
        if self._window.is_full:
            self._level = self._classifier(self._window.as_array())
        return self._state.update(self._level)

    @property
    def current_tier(self) -> AlertTier:
        return self._state.current_tier

    @property
    def fatigue_level(self) -> int:
        return self._level
