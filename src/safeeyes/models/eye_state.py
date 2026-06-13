"""Eye state classifier: a compact CNN that labels an eye crop open or closed.

The network is deliberately small so it can run in real time on the edge device
after quantization. An adaptive pooling layer makes the architecture independent
of the exact crop size, so the same model definition works whatever eye crop
resolution the perception stage settles on. Training happens on the laptop
against the MRL Eye split; this module is the architecture and the input
transform it expects.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

_POOLED = 4


class EyeStateCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((_POOLED, _POOLED))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * _POOLED * _POOLED, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.classifier(self.pool(self.features(x)))
        return logits


def preprocess_eye(image: np.ndarray, size: int = 24) -> torch.Tensor:
    """Turn an eye crop into a normalized single channel tensor of shape (1, size, size)."""
    import cv2

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0)
