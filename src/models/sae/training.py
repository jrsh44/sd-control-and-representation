import torch


def criterion_laux(x, x_hat, pre_codes, codes, dictionary):
    """
    Custom criterion function for Sparse Autoencoder (SAE) training.
    Args:
        x (torch.Tensor): Original input tensor.
        x_hat (torch.Tensor): Reconstructed output tensor from the SAE.
        pre_codes (torch.Tensor): Pre-activation codes from the encoder.
        codes (torch.Tensor): Final sparse codes after applying top-k sparsity.
        dictionary (torch.Tensor): The learned dictionary (decoder weights).
    n (int): Dimensionality of the input features.
    Returns:
        torch.Tensor: The computed loss value.
    """
    n = x.shape[1]

    # Compute reconstruction MSE
    mse = torch.mean((x - x_hat) ** 2)

    # Hyperparameters (these could be passed as arguments or globals)
    alpha = 1 / 32  # Scaling factor for auxiliary loss
    k_aux = n // 2  # Number of least active features for auxiliary loss

    # Compute mean activation per feature across the batch
    feature_acts = codes.mean(dim=0)

    # Select indices of k_aux features with the lowest mean activations (potential dead latents)
    low_act_indices = torch.topk(feature_acts, k_aux, largest=False)[1]

    # Create a masked codes tensor using only the low-activation features
    codes_aux = torch.zeros_like(codes)
    codes_aux[:, low_act_indices] = codes[:, low_act_indices]

    # Reconstruct using only these features
    x_aux = torch.matmul(codes_aux, dictionary)

    # Compute auxiliary MSE on this partial reconstruction
    l_aux = torch.mean((x - x_aux) ** 2)

    # Total metric: MSE + alpha * L_aux
    metric = mse + alpha * l_aux

    return metric
