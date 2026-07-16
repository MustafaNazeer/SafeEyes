"""Reference baselines for the distraction classifier.

Two baselines set the floor the pretrained backbones must clear. The majority
class predictor is model free: it learns nothing beyond the most common label in
the training records and predicts it for every frame, so its interval metrics
show what a constant guesser achieves on this split. The scratch CNN is a small
network trained from random initialization, mirroring the eye state architecture,
which isolates the contribution of transfer learning. Both are evaluated through
the same interval aggregation path as the real models.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from safeeyes.models.distraction_data import FrameRecord
from safeeyes.models.eval_distraction import N_CLASSES

_POOLED = 4


def fit_majority_class(records: Sequence[FrameRecord]) -> int:
    if not records:
        raise ValueError("cannot fit a majority class on an empty record sequence")
    counts = Counter(record.label_index for record in records)
    top = max(counts.values())
    return min(index for index, count in counts.items() if count == top)


def majority_frame_predictor(
    majority_index: int,
) -> Callable[[Sequence[Path]], np.ndarray]:
    def predict_frame_probs(frame_paths: Sequence[Path]) -> np.ndarray:
        probs = np.zeros((len(frame_paths), N_CLASSES), dtype=float)
        probs[:, majority_index] = 1.0
        return probs

    return predict_frame_probs


class DistractionScratchCNN(nn.Module):
    def __init__(self, num_classes: int = N_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((_POOLED, _POOLED))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * _POOLED * _POOLED, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.classifier(self.pool(self.features(x)))
        return logits
