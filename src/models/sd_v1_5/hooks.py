"""
Representation capture logic for Stable Diffusion v1.5.
Handles hook registration and activation capture during generation.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import StableDiffusionPipeline

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
    capture_latents: bool = False,
) -> tuple[List[torch.Tensor], Any] | tuple[List[torch.Tensor], Any, torch.Tensor]:
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
        capture_latents (bool): If True, also captures raw latent tensors [timesteps, batch, 4, 64, 64].

    Returns:
        If capture_latents=False:
            Tuple[List[torch.Tensor], Any]: (representations, image)
        If capture_latents=True:
            Tuple[List[torch.Tensor], Any, torch.Tensor]: (representations, image, latents)
        Where:
            - representations: List of captured activation tensors with shape [timesteps, ...], one per layer
            - image: Generated PIL Image
            - latents: Captured latents [timesteps, batch, 4, 64, 64] (only when capture_latents=True)
    """
    captured_representations: Dict[str, List[torch.Tensor]] = {
        f"hook_{i}": [] for i in range(len(layer_paths))
    }
    hook_handles: List[Any] = []

    do_classifier_free_guidance = guidance_scale > 1.0

    def create_capture_hook(name: str):
        """Factory function to create a hook that captures module output at each timestep."""

        def hook(model, input, output):
            if isinstance(output, tuple):
                tensor = output[0].detach().cpu()
            else:
                tensor = output.detach().cpu()

            if do_classifier_free_guidance:
                batch_size = tensor.shape[0]
                if batch_size == 2:
                    tensor = tensor[1:2]
                captured_representations[name].append(tensor)
            else:
                captured_representations[name].append(tensor)

        return hook

    for i, layer_enum in enumerate(layer_paths):
        path = str(layer_enum)
        hook_name = f"hook_{i}"

        try:
            module = get_nested_module(pipe, path)
            handle = module.register_forward_hook(create_capture_hook(hook_name))
            hook_handles.append(handle)
        except Exception as e:
            print(f"ERROR: Cannot find module for path '{path}'. Error: {e}")

    captured_latents_list = []

    if capture_latents:

        def latent_callback(pipe_obj, step_idx, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            if do_classifier_free_guidance and latents.shape[0] == 2:
                latents = latents[1:2]
            captured_latents_list.append(latents.cpu().clone())
            return callback_kwargs

        actual_callback = latent_callback
        actual_tensor_inputs = ["latents"]
    else:
        actual_callback = None
        actual_tensor_inputs = ["latents"]

    pipe_args = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "callback_on_step_end_tensor_inputs": actual_tensor_inputs,
    }

    if actual_callback is not None:
        pipe_args["callback_on_step_end"] = actual_callback

    image = pipe(**pipe_args).images[0]

    for handle in hook_handles:
        handle.remove()

    results = []
    for i in range(len(layer_paths)):
        hook_name = f"hook_{i}"
        if hook_name in captured_representations and captured_representations[hook_name]:
            timestep_tensors = captured_representations[hook_name]

            if (
                skip_initial_timestep
                and len(timestep_tensors) > 1
                and "unet." in str(layer_paths[i])
            ):
                timestep_tensors = timestep_tensors[1:]

            stacked = torch.stack(timestep_tensors, dim=0)  # [timesteps, batch, ...]
            print(f"Timestep: {i}, layer: {layer_paths[i].name}, stacked.shape: {stacked.shape}")
            if stacked.dim() == 5:
                t, b, h, w, f = stacked.shape
                stacked = stacked.reshape(t, b, h * w, f)

            results.append(stacked)
        else:
            results.append(None)

    if capture_latents:
        latents_tensor = None
        if captured_latents_list:
            latents_tensor = torch.stack(captured_latents_list, dim=0)
        return results, image, latents_tensor
    else:
        return results, image


def capture_layer_representations_with_unlearning(
    pipe: StableDiffusionPipeline,
    prompt: str,
    layer_paths: List[LayerPath],
    modifier: RepresentationModifier,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Generates an image with unlearning using a pre-configured RepresentationModifier.

    Args:
        pipe: Stable Diffusion pipeline
        prompt: text prompt for generation
        layer_paths: layers from which to capture activations
        modifier: pre-configured RepresentationModifier object (already set up with sae, means, top-N, etc.)
        ... rest of generation parameters

    Returns:
        (list_of_activations_per_layer, generated_image)
    """
    modifier.reset_timestep()

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
                x = x[1:2]
            captured[name].append(x)

        return hook

    for i, layer_path in enumerate(layer_paths):
        module = get_nested_module(pipe, str(layer_path.value))
        handle = module.register_forward_hook(make_capture_hook(f"hook_{i}"))
        capture_handles.append(handle)

    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    for h in capture_handles:
        h.remove()

    results = []
    for i in range(len(layer_paths)):
        tensors = captured[f"hook_{i}"]
        if not tensors:
            results.append(None)
            continue

        stacked = torch.stack(tensors, dim=0)

        if stacked.dim() == 5:
            t, b, h, w, c = stacked.shape
            stacked = stacked.reshape(t, b, h * w, c)

        results.append(stacked)

    return results, image
