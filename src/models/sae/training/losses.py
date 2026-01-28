"""
Loss functions for Sparse Autoencoder training.
"""

from typing import Tuple

import torch


def criterion_laux(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    pre_codes: torch.Tensor,
    codes: torch.Tensor,
    dictionary: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Top-K Auxiliary Loss

    This loss encourages dead features to revive by forcing them to predict
    the 'residual' that the main Top-K features failed to capture.

    Args:
        x: Original input (batch_size, input_dim).
        x_hat: Reconstructed output (batch_size, input_dim).
        pre_codes: Latent values BEFORE Top-K and ReLU (batch_size, dict_size).
        codes: Final sparse latent values AFTER Top-K (batch_size, dict_size).
        dictionary: The decoder weight matrix (dict_size, input_dim).

    Returns:
        Tuple of (reconstruction_loss, auxiliary_loss).
    """

    residual = x - x_hat
    main_mse = residual.square().mean()

    potential_activations = torch.relu(pre_codes)
    dead_features_val = potential_activations - codes

    aux_k = dictionary.shape[0] // 2
    aux_topk = torch.topk(dead_features_val, k=aux_k, dim=1)

    aux_codes = torch.zeros_like(codes)
    aux_codes.scatter_(-1, aux_topk.indices, aux_topk.values)

    residual_hat = aux_codes @ dictionary

    aux_mse = (residual.detach() - residual_hat).square().mean()

    return main_mse, aux_mse
