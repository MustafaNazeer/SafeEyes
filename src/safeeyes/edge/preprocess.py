"""Torch-free frame preprocessing for the distraction classifier on the edge.

The distraction backbone was trained with a torchvision eval transform, but the
Raspberry Pi runtime carries no torch or torchvision. This module reproduces that
transform with numpy and cv2 alone, so the same normalization the model learned
is applied on device. The constants are duplicated here rather than imported from
the training module (which pulls torchvision); a test pins them to the training
values so the two cannot drift apart.

The eval transform is, in order: BGR to RGB, resize to a square with INTER_AREA,
channels-first float scaled to [0, 1], then per-channel ImageNet normalization.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_DEFAULT_SIZE = 224


def preprocess_distraction_frame(
    frame_bgr: np.ndarray,
    size: int = _DEFAULT_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    normalized = (chw - mean_arr) / std_arr
    return normalized[None].astype(np.float32)
