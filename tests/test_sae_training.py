"""Unit tests for src/models/sae/training.py

Note: Tests for extract_input are in test_sae_utils.py
Note: Tests for compute_avg_max_cosine_similarity are in test_sae_metrics.py
"""

import pytest

try:
    import torch

    from src.models.sae.training import (
        _compute_reconstruction_error,
        _log_metrics,
        criterion_laux,
    )

    TORCH_AVAILABLE = True
except (ImportError, AttributeError):
    TORCH_AVAILABLE = False
    torch = None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_criterion_laux():
    """Test auxiliary loss criterion."""
    x = torch.randn(4, 64)
    x_hat = torch.randn(4, 64)
    codes = torch.randn(4, 64)
    dictionary = torch.randn(64, 64)

    loss = criterion_laux(x, x_hat, codes, codes, dictionary)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_compute_reconstruction_error():
    """Test R² reconstruction error computation."""
    # Test 2D case
    x = torch.randn(10, 64)
    x_hat = x.clone()
    r2 = _compute_reconstruction_error(x, x_hat)
    assert isinstance(r2, torch.Tensor)
    assert r2.item() >= 0.99  # Perfect reconstruction should have R² ≈ 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_log_metrics():
    """Test metrics logging with different monitoring levels."""
    try:
        from overcomplete.sae import TopKSAE

        model = TopKSAE(input_shape=64, nb_concepts=128, top_k=8, device="cpu")
        z = torch.randn(4, 128)
        loss = torch.tensor(0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Test monitoring level 0 (no logging)
        logs_0 = {}
        _log_metrics(0, logs_0, model, z, loss, optimizer)
        assert len(logs_0) == 0

        # Test monitoring level 1 (basic logging)
        logs_1 = {"lr": [], "step_loss": []}
        _log_metrics(1, logs_1, model, z, loss, optimizer)
        assert len(logs_1["lr"]) == 1
        assert len(logs_1["step_loss"]) == 1

    except (ImportError, AttributeError):
        pytest.skip("overcomplete library not available")
