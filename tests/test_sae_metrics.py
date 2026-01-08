"""Unit tests for src/models/sae/training/metrics.py"""

import pytest

try:
    import torch
    from src.models.sae.training.metrics import (
        compute_reconstruction_error,
        compute_sparsity_metrics,
        compute_dictionary_metrics,
        compute_avg_max_cosine_similarity,
    )

    TORCH_AVAILABLE = True
except (ImportError, AttributeError):
    TORCH_AVAILABLE = False
    torch = None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_compute_reconstruction_error_2d():
    """Test reconstruction error with 2D tensors."""
    x = torch.randn(10, 64)
    x_hat = x.clone()
    r2 = compute_reconstruction_error(x, x_hat)
    assert isinstance(r2, float)
    assert r2 >= 0.99  # Perfect reconstruction


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_compute_reconstruction_error_4d():
    """Test reconstruction error with 4D tensors (image-like)."""
    x = torch.randn(2, 3, 8, 8)  # (batch, channels, height, width)
    x_hat_flat = torch.randn(2 * 8 * 8, 3)  # Flattened reconstruction
    # This should work with shape mismatch handling
    r2 = compute_reconstruction_error(x, x_hat_flat)
    assert isinstance(r2, float)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_compute_reconstruction_error_3d():
    """Test reconstruction error with 3D tensors."""
    x = torch.randn(4, 10, 64)  # (batch, time, features)
    x_hat_flat = torch.randn(4 * 10, 64)  # Flattened reconstruction
    r2 = compute_reconstruction_error(x, x_hat_flat)
    assert isinstance(r2, float)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_compute_reconstruction_error_poor():
    """Test reconstruction error with poor reconstruction."""
    x = torch.randn(10, 64)
    x_hat = torch.randn(10, 64)  # Random, unrelated
    r2 = compute_reconstruction_error(x, x_hat)
    assert isinstance(r2, float)
    assert r2 < 1.0  # Should be worse than perfect


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_compute_sparsity_metrics_sparse():
    """Test sparsity metrics with sparse tensor."""
    # Create sparse tensor: only 10% non-zero
    z = torch.zeros(8, 100)
    z[:, :10] = torch.randn(8, 10)

    metrics = compute_sparsity_metrics(z)
    assert isinstance(metrics, dict)
    assert "l0_sparsity" in metrics
    assert "z_l2" in metrics
    assert "mean_activation" in metrics
    assert "max_activation" in metrics
    assert metrics["l0_sparsity"] <= 20  # Should be close to 10


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_compute_sparsity_metrics_dense():
    """Test sparsity metrics with dense tensor."""
    z = torch.randn(8, 100)
    metrics = compute_sparsity_metrics(z)
    assert metrics["l0_sparsity"] > 90  # Most values should be non-zero


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_compute_sparsity_metrics_all_zeros():
    """Test sparsity metrics with all zeros."""
    z = torch.zeros(8, 100)
    metrics = compute_sparsity_metrics(z)
    assert metrics["l0_sparsity"] == 0
    assert metrics["mean_activation"] == 0
    assert metrics["max_activation"] == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_compute_dictionary_metrics():
    """Test dictionary metrics computation."""
    dictionary = torch.randn(128, 64)
    metrics = compute_dictionary_metrics(dictionary)

    assert isinstance(metrics, dict)
    assert "sparsity" in metrics
    assert "norms_mean" in metrics
    assert isinstance(metrics["sparsity"], float)
    assert isinstance(metrics["norms_mean"], float)
    assert metrics["norms_mean"] > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_compute_avg_max_cosine_similarity_identical():
    """Test cosine similarity with identical rows."""
    weight_matrix = torch.ones(5, 10)
    result = compute_avg_max_cosine_similarity(weight_matrix)
    assert isinstance(result, float)
    assert result >= 0.99  # All rows identical -> high similarity


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_compute_avg_max_cosine_similarity_orthogonal():
    """Test cosine similarity with orthogonal vectors."""
    # Create orthogonal vectors
    weight_matrix = torch.eye(10)  # Identity matrix has orthogonal rows
    result = compute_avg_max_cosine_similarity(weight_matrix)
    assert isinstance(result, float)
    assert result < 0.1  # Orthogonal vectors should have low similarity


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_compute_avg_max_cosine_similarity_random():
    """Test cosine similarity with random matrix."""
    weight_matrix = torch.randn(50, 128)
    result = compute_avg_max_cosine_similarity(weight_matrix)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0  # Should be in valid range


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_compute_sparsity_metrics_inf_handling():
    """Test that infinite values are handled gracefully."""
    z = torch.tensor([[float("inf"), 1.0], [2.0, 3.0]])
    metrics = compute_sparsity_metrics(z)
    # Should not crash, metrics should be finite or zero
    assert isinstance(metrics["z_l2"], float)
    assert isinstance(metrics["mean_activation"], float)
