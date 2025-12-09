"""
Loss functions for Sparse Autoencoder training.
"""

from typing import Tuple

import torch


def criterion_laux(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    codes: torch.Tensor,
    dictionary: torch.Tensor,
    alpha: float = 1 / 32,
) -> torch.Tensor:
    """
    Custom criterion function for SAE training.

    Uses auxiliary loss to encourage activation of potentially dead features.
    The auxiliary loss reconstructs using only the least-active features,
    which helps prevent feature collapse and dead latents.

    Args:
        x: Original input tensor (batch_size, input_dim).
        x_hat: Reconstructed output tensor from the SAE.
        codes: Final sparse codes after applying top-k sparsity.
        dictionary: The learned dictionary (decoder weights).
        alpha: Scaling factor for auxiliary loss (default: 1/32).

    Returns:
        The computed loss value (MSE + alpha * auxiliary_loss).
    """
    recon_loss, aux_loss = criterion_laux_detailed(x, x_hat, codes, dictionary)
    return recon_loss + alpha * aux_loss


def criterion_laux_detailed(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    codes: torch.Tensor,
    dictionary: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom criterion function for SAE training with separate loss components.

    Uses auxiliary loss to encourage activation of potentially dead features.
    The auxiliary loss reconstructs using only the least-active features,
    which helps prevent feature collapse and dead latents.

    Args:
        x: Original input tensor (batch_size, input_dim).
        x_hat: Reconstructed output tensor from the SAE.
        codes: Final sparse codes after applying top-k sparsity.
        dictionary: The learned dictionary (decoder weights).

    Returns:
        Tuple of (reconstruction_loss, auxiliary_loss).
    """
    n = x.shape[1]

    # Compute reconstruction MSE
    recon_loss = torch.mean((x - x_hat) ** 2)

    # Number of least active features for auxiliary loss
    k_aux = n // 2

    # Compute mean activation per feature across the batch
    feature_acts = codes.mean(dim=0)

    # Select indices of k_aux features with the lowest mean activations
    low_act_indices = torch.topk(feature_acts, k_aux, largest=False)[1]

    # Create a masked codes tensor using only the low-activation features
    codes_aux = torch.zeros_like(codes)
    codes_aux[:, low_act_indices] = codes[:, low_act_indices]

    # Reconstruct using only these features
    x_aux = torch.matmul(codes_aux, dictionary)

    # Compute auxiliary MSE on this partial reconstruction
    aux_loss = torch.mean((x - x_aux) ** 2)

    return recon_loss, aux_loss
