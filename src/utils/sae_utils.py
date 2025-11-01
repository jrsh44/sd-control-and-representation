import torch
from einops import rearrange
from overcomplete.sae import TopKSAE, train_sae
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

""" Note:
    The global variables:
    - sae
    - concept_vector
    - step_images
    - pipe
    - sae_encodings
    are assumed to be defined elsewhere.
"""

# --- 1. SAE training functions ---


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

# --- 2. SAE feature selection ---


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


# --- 3. SAE integration with Stable Diffusion ---


def callback_handler(self, step: int, timestep: int, callback_kwargs: dict):
    global step_images, pipe

    try:
        latents = callback_kwargs["latents"]
        latent_to_decode = latents[0].detach()
        latent_to_decode = 1 / pipe.vae.config.scaling_factor * latent_to_decode

        with torch.no_grad():
            image = pipe.vae.decode(latent_to_decode.unsqueeze(0)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).astype("uint8"))

        step_images.append(image)
        print(f"Timestep {timestep}")
        return {}
    except Exception as e:
        print(f"Callback error: {e}")
        return {}


def latent_modification(codes: torch.Tensor) -> torch.Tensor:
    """
    Modyfikuje wybrane koncepcje w reprezentacji latentnej.

    Args:
        codes (torch.Tensor): Tensor kodów latentnych o wymiarach (batch_size, num_concepts).
    Returns:
        torch.Tensor: Zmodyfikowany tensor kodów latentnych.
    """
    global concept_vector
    modified_codes = codes - 100 * (codes * concept_vector)  # Zero out selected concepts
    return modified_codes


def sae_integration_hook(module, input, output):
    """Hook integrujący SAE z modelem U-Net.
    args:
        module: Moduł, do którego jest podłączony hook.
        input: Wejście do modułu (nieużywane tutaj).
        output: Wyjście z modułu (aktywacje U-Net).
    """
    try:
        global sae_encodings, concept_vector, sae
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output
        original_dtype = act.dtype  # Should be torch.float16
        act = act.float()
        batch, seq_len, dim = act.shape
        act_reshaped = rearrange(act, "b t d -> (b t) d")
        with torch.no_grad():
            pre_codes, codes = sae.encode(act_reshaped)  # sae on 'cuda'
            modified_codes = latent_modification(codes)
            act_reconstructed = sae.decode(modified_codes)
            act_reconstructed = rearrange(act_reconstructed, "(b t) d -> b t d", b=batch, t=seq_len)
            act_reconstructed = act_reconstructed.to(dtype=original_dtype)
        sae_encodings.append(
            {
                "pre_codes": pre_codes,  # Keep on GPU
                "codes": codes,
                "modified_codes": modified_codes,
            }
        )
        if isinstance(output, tuple):
            return (act_reconstructed,) + output[1:]
        return act_reconstructed
    except Exception as e:
        print(f"Hook error: {e}")
        raise
