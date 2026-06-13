"""Temporal fatigue classifiers.

Two models map a window to a fatigue level (alert, low vigilance, drowsy). The
GRU consumes the raw per frame feature sequence and is the primary model; the
gradient boosted trees baseline consumes the aggregated window features and
exists as an honest, simple point of comparison. Both are trained and evaluated
on the same subject independent UTA-RLDD split.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from torch import nn


class TemporalGRU(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        num_classes: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            n_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        logits: torch.Tensor = self.head(hidden[-1])
        return logits


class GBTBaseline:
    def __init__(self, random_state: int = 0, **kwargs: Any) -> None:
        self._clf = GradientBoostingClassifier(random_state=random_state, **kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray) -> GBTBaseline:
        self._clf.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._clf.predict(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._clf.predict_proba(x))
