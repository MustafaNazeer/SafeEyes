import numpy as np
import torch

from safeeyes.models.eye_state import EyeStateCNN, preprocess_eye


def test_forward_returns_logits_per_class() -> None:
    model = EyeStateCNN(num_classes=2)
    out = model(torch.zeros(8, 1, 24, 24))
    assert out.shape == (8, 2)


def test_forward_is_input_size_agnostic() -> None:
    model = EyeStateCNN(num_classes=2)
    assert model(torch.zeros(4, 1, 32, 32)).shape == (4, 2)
    assert model(torch.zeros(4, 1, 48, 48)).shape == (4, 2)


def test_model_has_trainable_parameters() -> None:
    model = EyeStateCNN()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0


def test_training_reduces_loss_on_a_fixed_batch() -> None:
    torch.manual_seed(0)
    model = EyeStateCNN(num_classes=2)
    x = torch.randn(16, 1, 24, 24)
    y = torch.randint(0, 2, (16,))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    first = criterion(model(x), y).item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    last = criterion(model(x), y).item()
    assert last < first


def test_preprocess_eye_grayscale_resized_normalized() -> None:
    image = np.full((40, 30, 3), 255, dtype=np.uint8)
    tensor = preprocess_eye(image, size=24)
    assert tensor.shape == (1, 24, 24)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_preprocess_eye_accepts_single_channel() -> None:
    image = np.zeros((40, 30), dtype=np.uint8)
    tensor = preprocess_eye(image, size=16)
    assert tensor.shape == (1, 16, 16)
