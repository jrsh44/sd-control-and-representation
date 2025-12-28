"""
Metrics for Sparse Autoencoder training and evaluation.
"""

import math

import torch
from einops import rearrange
from overcomplete.metrics import l0_eps, l2, r2_score


def compute_reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """
    Compute reconstruction error as R² score.

    Handles different input shapes by flattening appropriately:
    - 4D input (n, c, w, h): flattens to (n*w*h, c)
    - 3D input (n, t, c): flattens to (n*t, c)
    - 2D input: uses directly

    Args:
        x: Original input tensor
        x_hat: Reconstructed tensor

    Returns:
        R² score (1.0 = perfect reconstruction, 0.0 = no reconstruction)
    """
    if len(x.shape) == 4 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n c w h -> (n w h) c")
    elif len(x.shape) == 3 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n t c -> (n t) c")
    else:
        if x.shape != x_hat.shape:
            raise ValueError("Input and output shapes must match.")
        x_flatten = x

    r2 = r2_score(x_flatten, x_hat)
    return r2.item()


def compute_sparsity_metrics(z: torch.Tensor) -> dict:
    """
    Compute sparsity-related metrics for encoded representations.

    Args:
        z: Encoded tensor (batch_size, num_features)

    Returns:
        Dictionary with:
        - l0_sparsity: Average L0 norm per sample (using overcomplete.metrics)
        - z_l2: L2 norm of activations (average across batch)
        - mean_activation: Mean activation value
        - max_activation: Maximum activation value
    """
    epsilon = 1e-8

    is_active = (z.abs() > epsilon).float()
    l0_per_sample = is_active.sum(dim=-1).mean().item()

    z_norms = torch.norm(z, p=2, dim=-1)
    z_l2 = z_norms.mean().item()

    mean_act = z.abs().mean().item()
    max_act = z.abs().max().item()

    if not math.isfinite(z_l2):
        z_l2 = 0.0
    if not math.isfinite(mean_act):
        mean_act = 0.0

    return {
        "l0_sparsity": l0_per_sample,
        "z_l2": z_l2,
        "mean_activation": mean_act,
        "max_activation": max_act,
    }


def compute_dictionary_metrics(dictionary: torch.Tensor) -> dict:
    """
    Compute metrics for the learned dictionary (decoder weights).

    Args:
        dictionary: Dictionary tensor (num_features, input_dim)

    Returns:
        Dictionary with:
        - sparsity: Average sparsity of dictionary columns
        - norms_mean: Mean L2 norm of dictionary columns
    """
    return {
        "sparsity": l0_eps(dictionary).mean().item(),
        "norms_mean": l2(dictionary, dims=-1).mean().item(),
    }


def compute_avg_max_cosine_similarity(weight_matrix: torch.Tensor) -> float:
    """
    Compute the average maximum cosine similarity for a weight matrix.

    For each feature (row) in the weight matrix, computes its cosine similarity
    with all other features, finds the maximum similarity (excluding self),
    and returns the average of these maximum similarities.

    This metric helps understand feature redundancy - higher values indicate
    more similar/redundant features in the dictionary.

    Args:
        weight_matrix: Weight matrix of shape (nb_features, feature_dim)

    Returns:
        Average maximum cosine similarity across all features
    """
    # Normalize weights to unit norm (L2 normalize each row)
    normalized = torch.nn.functional.normalize(weight_matrix, p=2, dim=1)

    # Compute absolute pairwise cosine similarities (nb_features x nb_features)
    cos_sim = (normalized @ normalized.T).abs()

    # Set diagonal to -inf to exclude self-similarity when finding max
    cos_sim.fill_diagonal_(-float("inf"))

    # For each feature, find max similarity with other features, then average
    avg_max_cos = cos_sim.max(dim=1)[0].mean().item()

    return avg_max_cos


def compute_encoder_decoder_similarity(model: torch.nn.Module) -> dict:
    """
    Compute cosine similarity metrics for encoder and decoder weights.

    Args:
        model: SAE model with encoder and decoder

    Returns:
        Dictionary with:
        - encoder_avg_max_cos: Average max cosine similarity between encoder weights
        - decoder_avg_max_cos: Average max cosine similarity between decoder weights
        - decoder_mean_norm: Average norm of decoder vectors
    """
    metrics = {}

    # Get decoder weights
    dictionary = model.get_dictionary
    if callable(dictionary):
        dictionary = dictionary()

    metrics["decoder_avg_max_cos"] = compute_avg_max_cosine_similarity(dictionary)
    metrics["decoder_mean_norm"] = torch.norm(dictionary, dim=1).mean().item()

    # Try to get encoder weights (TopKSAE specific)
    try:
        encoder_weights = model.encoder.final_block[0].weight.data  # type: ignore
        metrics["encoder_avg_max_cos"] = compute_avg_max_cosine_similarity(encoder_weights)
    except (AttributeError, IndexError, TypeError):
        # Model doesn't have expected encoder structure
        metrics["encoder_avg_max_cos"] = 0.0

    return metrics


def compute_gradient_metrics(model: torch.nn.Module) -> dict:
    """
    Compute gradient statistics for model parameters.

    Args:
        model: The SAE model

    Returns:
        Dictionary with gradient norms for each parameter
    """
    metrics = {}
    for name, param in model.named_parameters():
        param_norm = l2(param).item()
        metrics[f"param_norm/{name}"] = param_norm

        if param.grad is not None:
            grad_norm = l2(param.grad).item()
            metrics[f"grad_norm/{name}"] = grad_norm

            if param_norm > 0:
                metrics[f"grad_weight_ratio/{name}"] = grad_norm / param_norm

    return metrics


class MetricsAggregator:
    """
    Aggregates metrics over batches for epoch-level reporting.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self._sums: dict = {}
        self._counts: dict = {}
        self._last_values: dict = {}

    def update(self, metrics: dict, count: int = 1):
        """
        Update running sums with new batch metrics.

        Args:
            metrics: Dictionary of metric values
            count: Number of samples in this batch
        """
        for key, value in metrics.items():
            if key not in self._sums:
                self._sums[key] = 0.0
                self._counts[key] = 0
            self._sums[key] += value * count
            self._counts[key] += count
            self._last_values[key] = value

    def get_averages(self) -> dict:
        """Get averaged metrics over all updates."""
        return {
            key: self._sums[key] / self._counts[key] for key in self._sums if self._counts[key] > 0
        }

    def get_last(self) -> dict:
        """Get the last recorded values."""
        return self._last_values.copy()

    def get_sums(self) -> dict:
        """Get raw sums."""
        return self._sums.copy()
