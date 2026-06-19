import numpy as np
import torch

from safeeyes.temporal.model import GBTBaseline, TemporalGRU


def test_gru_forward_returns_logits_per_class() -> None:
    model = TemporalGRU(n_features=5, num_classes=3)
    out = model(torch.zeros(8, 10, 5))
    assert out.shape == (8, 3)


def test_gru_handles_variable_sequence_length() -> None:
    model = TemporalGRU(n_features=5, num_classes=3)
    assert model(torch.zeros(4, 25, 5)).shape == (4, 3)


def test_gru_training_reduces_loss_on_fixed_batch() -> None:
    torch.manual_seed(0)
    model = TemporalGRU(n_features=4, num_classes=3)
    x = torch.randn(16, 8, 4)
    y = torch.randint(0, 3, (16,))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = criterion(model(x), y).item()
    for _ in range(40):
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    assert criterion(model(x), y).item() < first


def test_gru_default_normalization_is_identity() -> None:
    model = TemporalGRU(n_features=3, num_classes=3)
    assert torch.allclose(model.feature_mean, torch.zeros(3))
    assert torch.allclose(model.feature_std, torch.ones(3))


def test_gru_normalization_matches_prestandardized_input() -> None:
    torch.manual_seed(0)
    model = TemporalGRU(n_features=2, num_classes=3)
    mean = torch.tensor([1.0, 10.0])
    std = torch.tensor([2.0, 5.0])
    raw = torch.randn(4, 6, 2) * std + mean
    standardized = (raw - mean) / std
    model.eval()
    with torch.no_grad():
        out_identity = model(standardized)  # default identity normalization
        model.set_normalization(mean, std)
        out_normalized = model(raw)  # normalizes internally to the same values
    assert torch.allclose(out_identity, out_normalized, atol=1e-5)


def test_gru_set_normalization_survives_zero_std() -> None:
    model = TemporalGRU(n_features=2, num_classes=3)
    model.set_normalization(mean=[0.0, 0.0], std=[0.0, 1.0])
    out = model(torch.randn(1, 4, 2))
    assert torch.all(torch.isfinite(out))


def test_gbt_baseline_overfits_separable_data() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.normal(-2.0, 0.1, size=(20, 3))
    x1 = rng.normal(2.0, 0.1, size=(20, 3))
    x = np.vstack([x0, x1])
    y = np.array([0] * 20 + [1] * 20)
    clf = GBTBaseline(random_state=0)
    clf.fit(x, y)
    assert (clf.predict(x) == y).mean() == 1.0


def test_gbt_predict_proba_shape() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 4))
    y = rng.integers(0, 3, size=30)
    clf = GBTBaseline(random_state=0)
    clf.fit(x, y)
    proba = clf.predict_proba(x)
    assert proba.shape == (30, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
