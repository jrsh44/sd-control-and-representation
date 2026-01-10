"""
Representation capture logic for Stable Diffusion v1.5.
Handles hook registration and activation capture during generation.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import StableDiffusionPipeline

# from einops import rearrange
from src.utils.RepresentationModifier import RepresentationModifier

from .layers import LayerPath


def get_nested_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Retrieves a nested module from a model using a dot-separated path string.

    Args:
        model (torch.nn.Module): The root model to navigate.
        path (str): Dot-separated path to the target module (e.g., "unet.down_blocks.1.resnets.0").

    Returns:
        torch.nn.Module: The target module at the specified path.
    """
    if not path:
        return model

    path_segments = path.split(".")
    current_module = model
    for name in path_segments:
        if name.isdigit() or name.startswith("-") and name[1:].isdigit():
            current_module = current_module[int(name)]
        else:
            current_module = getattr(current_module, name)

    return current_module


def capture_layer_representations(
    pipe: StableDiffusionPipeline,
    prompt: str,
    layer_paths: List[LayerPath],
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: torch.Generator = None,
    skip_initial_timestep: bool = False,
) -> tuple[List[torch.Tensor], Any]:
    """
    Captures intermediate representations from specified layers during SD generation.
    Collects representations for ALL timesteps during denoising process.

    Args:
        pipe (StableDiffusionPipeline): The Stable Diffusion pipeline instance.
        prompt (str): Text prompt for image generation.
        layer_paths (List[LayerPath]): List of layer paths to capture representations from.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Classifier-free guidance scale.
        generator (torch.Generator): Random generator for reproducibility.
        skip_initial_timestep (bool): If True, skips the very first captured representation
                                      for U-Net layers (the initial noisy latent).

    Returns:
        Tuple[List[torch.Tensor], Any]:
            - List of captured activation tensors with shape [timesteps, ...], one per layer
            - Generated PIL Image
    """
    # Store representations for each timestep: {hook_name: [list of tensors per timestep]}
    captured_representations: Dict[str, List[torch.Tensor]] = {
        f"hook_{i}": [] for i in range(len(layer_paths))
    }
    hook_handles: List[Any] = []

    # Check if classifier-free guidance is enabled
    do_classifier_free_guidance = guidance_scale > 1.0

    def create_capture_hook(name: str):
        """Factory function to create a hook that captures module output at each timestep."""

        def hook(model, input, output):
            if isinstance(output, tuple):
                tensor = output[0].detach().cpu()
            else:
                tensor = output.detach().cpu()

            # Handle classifier-free guidance: split batch and keep only conditional
            if do_classifier_free_guidance:
                # With CFG, batch contains [unconditional, conditional]
                # We only want the conditional part (second half)
                batch_size = tensor.shape[0]
                if batch_size == 2:
                    # Keep batch dim: [1, ...]
                    tensor = tensor[1:2]
                captured_representations[name].append(tensor)
            else:
                # No CFG, capture as-is
                captured_representations[name].append(tensor)

        return hook

    # Register forward hooks on target layers
    for i, layer_enum in enumerate(layer_paths):
        path = str(layer_enum)
        hook_name = f"hook_{i}"

        try:
            module = get_nested_module(pipe, path)
            handle = module.register_forward_hook(create_capture_hook(hook_name))
            hook_handles.append(handle)
        except Exception as e:
            print(f"ERROR: Cannot find module for path '{path}'. Error: {e}")

    # Generate image to trigger activations
    pipe_args = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }

    image = pipe(**pipe_args).images[0]

    # Clean up hooks
    for handle in hook_handles:
        handle.remove()

    # Stack timesteps and flatten spatial dimensions: [timesteps, batch, spatial, features]
    results = []
    for i in range(len(layer_paths)):
        hook_name = f"hook_{i}"
        if hook_name in captured_representations and captured_representations[hook_name]:
            timestep_tensors = captured_representations[hook_name]

            # Apply skip_initial_timestep logic for U-Net related layers
            # Text encoder layers (TEXT_EMBEDDING_FINAL etc.) only have 1 timestep
            # so this check should only apply to layers that might have > 1 timestep
            if (
                skip_initial_timestep
                and len(timestep_tensors) > 1
                and "unet." in str(layer_paths[i])
            ):
                timestep_tensors = timestep_tensors[1:]

            # Stack all timesteps into first dimension
            stacked = torch.stack(timestep_tensors, dim=0)  # [timesteps, batch, ...]

            # Flatten spatial dimensions if present
            if stacked.dim() == 5:
                # [timesteps, batch, h, w, features] -> [timesteps, batch, h*w, features]
                t, b, h, w, f = stacked.shape
                stacked = stacked.reshape(t, b, h * w, f)

            results.append(stacked)
        else:
            results.append(None)

    return results, image


def capture_layer_representations_with_unlearning(
    pipe: StableDiffusionPipeline,
    prompt: str,
    layer_paths: List[LayerPath],
    modifier: RepresentationModifier,  # ← gotowy modyfikator!
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Generuje obraz z unlearningiem przy użyciu gotowego RepresentationModifier.
    Przechwytuje aktywacje ze wszystkich timestepów z wybranych warstw.

    Args:
        pipe: Stable Diffusion pipeline
        prompt: tekstowy prompt
        layer_paths: warstwy do przechwytywania aktywacji
        modifier: gotowy obiekt RepresentationModifier (już skonfigurowany z sae, means, top-N itd.)
        ... reszta parametrów generowania

    Returns:s
        (list_of_activations_per_layer, generated_image)
    """
    # === 0. Zresetowanie modyfikatora (indeks timestepów) ===
    modifier.reset_timestep()

    # === 1. Hooki do przechwytywania aktywacji (wszystkie warstwy) ===
    captured = {f"hook_{i}": [] for i in range(len(layer_paths))}
    capture_handles = []

    do_cfg = guidance_scale > 1.0

    def make_capture_hook(name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            x = x.detach().cpu()
            if do_cfg and x.shape[0] == 2:
                x = x[1:2]  # tylko conditional (CFG)
            captured[name].append(x)

        return hook

    # Rejestrujemy hooki przechwytywania
    for i, layer_path in enumerate(layer_paths):
        module = get_nested_module(pipe, str(layer_path.value))
        handle = module.register_forward_hook(make_capture_hook(f"hook_{i}"))
        capture_handles.append(handle)

    # # === 2. Podłączamy modyfikator (on sam wie, do której warstwy się podpiąć) ===
    # with modifier:
    #     # === 3. Generowanie obrazu (modyfikator działa tylko w tym bloku) ===
    #     image = pipe(
    #         prompt=prompt,
    #         num_inference_steps=num_inference_steps,
    #         guidance_scale=guidance_scale,
    #         generator=generator,
    #     ).images[0]
    # === 2. Modifier is already attached externally, no need for context manager ===
    # === 3. Generowanie obrazu (modyfikator działa) ===
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # === 4. Usuwamy hooki przechwytywania ===
    for h in capture_handles:
        h.remove()

    # === 5. Przygotowanie wyników ===
    results = []
    for i in range(len(layer_paths)):
        tensors = captured[f"hook_{i}"]
        if not tensors:
            results.append(None)
            continue

        stacked = torch.stack(tensors, dim=0)  # [timesteps, ...]

        # Spłaszcz wymiary przestrzenne jeśli są
        if stacked.dim() == 5:  # [T, B, H, W, C]
            t, b, h, w, c = stacked.shape
            stacked = stacked.reshape(t, b, h * w, c)

        results.append(stacked)

    return results, image
