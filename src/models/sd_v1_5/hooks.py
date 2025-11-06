"""
Representation capture logic for Stable Diffusion v1.5.
Handles hook registration and activation capture during generation.
"""

from typing import Any, Dict, List

import torch
from diffusers import StableDiffusionPipeline

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
    current_timestep = [0]  # Mutable counter for timestep tracking

    def create_capture_hook(name: str):
        """Factory function to create a hook that captures module output at each timestep."""

        def hook(model, input, output):
            if isinstance(output, tuple):
                tensor = output[0].detach().cpu()
            else:
                tensor = output.detach().cpu()

            # Append this timestep's representation
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

    # Stack timesteps into single tensor per layer: [timesteps, ...]
    results = []
    for i in range(len(layer_paths)):
        hook_name = f"hook_{i}"
        if hook_name in captured_representations and captured_representations[hook_name]:
            # Stack all timesteps into first dimension
            timestep_tensors = captured_representations[hook_name]
            stacked = torch.stack(timestep_tensors, dim=0)  # [timesteps, batch, ...]
            results.append(stacked)
        else:
            results.append(None)

    return results, image
