import torch
from einops import rearrange
from PIL import Image


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
