"""Offline replay of the live decision path over recorded feature sequences.

The live pipeline computes a fatigue level per frame and feeds it to the alert
state machine. Replaying splits that in two: the classifier pass over a clip is
computed once (it does not depend on state machine parameters), and the cheap
state machine is then run over the cached levels, which is what makes an honest
parameter sweep affordable. Semantics match DrowsinessPipeline exactly: the
level is 0 until the rolling window first fills, then updates every frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from safeeyes.alert.pipeline import Classifier
from safeeyes.alert.state_machine import AlertStateMachine, AlertTier
from safeeyes.temporal.window import FeatureWindow


def classify_sequence(
    features: np.ndarray, classifier: Classifier, window_size: int = 150
) -> np.ndarray:
    feats = np.asarray(features, dtype=float)
    levels = np.zeros(feats.shape[0], dtype=int)
    if feats.shape[0] == 0:
        return levels
    window = FeatureWindow(window_size, feats.shape[1])
    level = 0
    for i in range(feats.shape[0]):
        window.push(feats[i])
        if window.is_full:
            level = classifier(window.as_array())
        levels[i] = level
    return levels


@dataclass(frozen=True)
class TierEvent:
    frame: int
    tier: AlertTier


def replay_levels(levels: Sequence[int], machine: AlertStateMachine) -> list[TierEvent]:
    machine.reset()
    events: list[TierEvent] = []
    last = AlertTier.NONE
    for i, level in enumerate(levels):
        tier = machine.update(int(level))
        if tier != last:
            events.append(TierEvent(frame=i, tier=tier))
            last = tier
    return events
