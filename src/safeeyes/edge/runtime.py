"""ONNX Runtime inference for the edge device.

A thin session wrapper plus two adapters that match the classifier contracts the
rest of the system already speaks. The window classifier has the same signature
as ``make_gru_classifier`` in the pipeline, so an exported and quantized model
drops into ``DrowsinessPipeline`` with no other change and no PyTorch on the Pi.
The eye classifier turns a single grayscale crop into an open/closed label. Both
adapters are verified to make the same decisions as the PyTorch reference.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxModel:
    def __init__(self, path: str | Path) -> None:
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def run(self, array: np.ndarray) -> np.ndarray:
        outputs = self._session.run(None, {self._input_name: array.astype(np.float32)})
        return np.asarray(outputs[0])


def make_onnx_window_classifier(model: OnnxModel) -> Callable[[np.ndarray], int]:
    def classify(window: np.ndarray) -> int:
        logits = model.run(np.asarray(window)[None])
        return int(logits.argmax(axis=1)[0])

    return classify


def make_onnx_eye_classifier(model: OnnxModel) -> Callable[[np.ndarray], int]:
    def classify(crop: np.ndarray) -> int:
        array = np.asarray(crop, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, None]
        elif array.ndim == 3:
            array = array[None]
        logits = model.run(array)
        return int(logits.argmax(axis=1)[0])

    return classify
