import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from overcomplete.sae import TopKSAE


def select_features(
    activations_concept_true_tensor, activations_concept_false_tensor, n_features, sae, epsilon=1e-8
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

    # Compute sums of activation on each feature
    sum_code_means_true = codes_true.sum(dim=0)
    sum_code_means_false = codes_false.sum(dim=0)

    normalized_diff = (
        sum_code_means_true / (sum_code_means_true + epsilon * sum_code_means_true.shape[0])
    ) - (sum_code_means_false / (sum_code_means_false + epsilon * sum_code_means_false.shape[0]))

    # Sort and select top n_features
    feature_indices = torch.argsort(normalized_diff, descending=True)
    selected_features = feature_indices[:n_features].tolist()

    return selected_features


def infer_sae_config(sae: TopKSAE):
    """Extract all config from the model itself."""
    enc_weight = sae.encoder.weight

    input_dim = enc_weight.shape[0]
    nb_concepts = enc_weight.shape[1]
    expansion_factor = nb_concepts // input_dim

    # top_k is stored as a buffer or attribute
    top_k = sae.top_k.item() if isinstance(sae.top_k, torch.Tensor) else sae.top_k

    return {
        "input_dim": input_dim,
        "nb_concepts": nb_concepts,
        "expansion_factor": expansion_factor,
        "top_k": top_k,
    }


# ------------------------------------------------------------------
# 2. SAE forward → middle-layer activations (the “codes”)
# ------------------------------------------------------------------
@torch.no_grad()
def get_codes(sae, x, device):
    _, z, _ = sae(x.to(device))
    return z


def concept_filtering_function(
    concept_name: str,
    concept_value: str,
    negate: bool = False,
) -> callable:
    """
    Creates a filtering function for the dataset based on concept name and value.
    like  (e.g., lambda x: x['object'] == 'cat' and x['style'] == 'Impressionism')

    Args:
        concept_name (str): Name of the concept (metadata column).
        concept_value (str): Value of the concept to filter by.
    Returns:
        function: A filtering function that can be used with the dataset's filter method.
    """
    if negate:
        return lambda x: x[concept_name] != concept_value
    else:
        return lambda x: x[concept_name] == concept_value


def compute_sums(loader, sae, device, nb_concepts):
    total_sum = torch.zeros(nb_concepts, dtype=torch.float64, device="cpu")

    for i, batch in enumerate(loader):
        codes = get_codes(sae, batch, device)
        codes_cpu = codes.to("cpu", dtype=torch.float64)
        total_sum += codes_cpu.sum(dim=0)

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(loader)} batches", end="\r")
    return total_sum
