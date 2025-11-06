import torch


def select_features(
    activations_concept_true_tensor, activations_concept_false_tensor, n_features, sae
):
    """
    Wybiera cechy o największej różnicy średnich wartości między dwiema grupami aktywacji.

    Args:
        activations_concept_true_tensor (torch.Tensor):
            Tensor aktywacji dla grupy "true" o kształcie [num_images, num_samples, num_d_maps].
        activations_concept_false_tensor (torch.Tensor):
            Tensor aktywacji dla grupy "false" o kształcie [num_images, num_samples, num_d_maps].
        n_features (int): Liczba cech do wybrania.

    Returns:
        List[int]: Lista indeksów wybranych cech.
    """
    # Move tensors to the same device as the SAE model
    device = next(sae.parameters()).device
    activations_concept_true_tensor = activations_concept_true_tensor.to(device, non_blocking=True)
    activations_concept_false_tensor = activations_concept_false_tensor.to(
        device, non_blocking=True
    )

    num_d_maps = activations_concept_true_tensor.shape[2]

    # Reshape and encode once for all features
    true_flat = activations_concept_true_tensor.reshape(-1, num_d_maps)
    false_flat = activations_concept_false_tensor.reshape(-1, num_d_maps)
    print(f"Encoding {true_flat.shape[0]} true samples and {false_flat.shape[0]} false samples...")
    codes_true = sae.encode(true_flat)[1]
    codes_false = sae.encode(false_flat)[1]

    # Compute sums of means
    sum_code_means_true = codes_true.sum(dim=0)
    sum_code_means_false = codes_false.sum(dim=0)

    # Vectorized computation of mean differences
    mean_true = codes_true.mean(dim=0)
    mean_false = codes_false.mean(dim=0)
    normalized_diff = (mean_true / (sum_code_means_true + 1e-8)) - (
        mean_false / (sum_code_means_false + 1e-8)
    )

    # Sort and select top n_features
    feature_indices = torch.argsort(normalized_diff, descending=True)
    selected_features = feature_indices[:n_features].tolist()

    return selected_features
