"""The edge distraction preprocess must reproduce the training eval transform.

On the Pi there is no torchvision, so the cabin frame goes through a numpy and
cv2 reimplementation of the eval transform. If that drifts from what the model
was trained on, every distraction prediction is silently wrong. These tests pin
the numpy path to the torchvision path byte close on a fixed image, and pin the
normalization constants to the ones training used.
"""

from __future__ import annotations

import numpy as np

from safeeyes.edge.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_distraction_frame,
)


def test_matches_torchvision_eval_transform() -> None:
    from safeeyes.models.train_distraction import build_transform

    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    expected = build_transform(train=False, size=224, normalize=True)(image).numpy()[None]

    result = preprocess_distraction_frame(image, size=224)

    assert result.shape == (1, 3, 224, 224)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_default_size_is_the_backbone_size() -> None:
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)
    result = preprocess_distraction_frame(image)
    assert result.shape == (1, 3, 224, 224)


def test_constants_match_training() -> None:
    from safeeyes.models.train_distraction import (
        IMAGENET_MEAN as TRAIN_MEAN,
    )
    from safeeyes.models.train_distraction import (
        IMAGENET_STD as TRAIN_STD,
    )

    assert IMAGENET_MEAN == TRAIN_MEAN
    assert IMAGENET_STD == TRAIN_STD
