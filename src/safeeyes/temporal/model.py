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
        # Feature normalization is part of the model so the exported graph takes
        # raw per frame features and standardizes them internally. The buffers are
        # saved in the checkpoint and travel to the edge runtime, keeping training
        # and inference consistent. They default to the identity transform.
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_std", torch.ones(n_features))
        self.gru = nn.GRU(
            n_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def set_normalization(self, mean: object, std: object) -> None:
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        std_t = torch.as_tensor(std, dtype=torch.float32)
        std_t = torch.where(std_t < 1e-6, torch.ones_like(std_t), std_t)
        self.feature_mean = mean_t
        self.feature_std = std_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feature_mean) / self.feature_std
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
