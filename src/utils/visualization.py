from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Image as IPImage
from IPython.display import display
from PIL import Image


def display_image(image: Image.Image, title: str = ""):
    """
    Displays a single PIL.Image object.

    Args:
        image (Image.Image): PIL.Image object to display.
        title (str): Plot title.
    """
    if not isinstance(image, Image.Image):
        print("Error: Expected a single PIL.Image object.")
        return

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.show()


def display_sequence(
    images: List[Image.Image],
    title: str = "",
    max_cols: int = 5,
    sampling_rate: int = 1,
):
    """
    Displays a sequence of PIL.Image objects in a grid, showing temporal progression.

    Args:
        images (List[Image.Image]): List of PIL.Image objects (generation steps).
        title (str): Main plot title.
        max_cols (int): Maximum number of columns in the grid.
        sampling_rate (int): Display every N-th image from the list to shorten the sequence.
    """
    if not isinstance(images, list) or not all(isinstance(img, Image.Image) for img in images):
        print("Error: Expected a list of PIL.Image objects.")
        return
    if not images:
        print("Image list is empty.")
        return

    display_list = []
    display_indices = []

    for i in range(0, len(images), sampling_rate):
        display_list.append(images[i])
        display_indices.append(i)

    # Ensure the last step is always visible
    last_idx = len(images) - 1
    if last_idx not in display_indices:
        display_list.append(images[last_idx])
        display_indices.append(last_idx)

    # --- Grid Configuration ---
    num_to_display = len(display_list)
    cols = min(num_to_display, max_cols)
    rows = (num_to_display + cols - 1) // cols

    plt.figure(figsize=(3 * cols, 3 * rows))
    plt.suptitle(title, fontsize=16)

    # --- Draw Images in Grid ---
    for i, (img, original_idx) in enumerate(zip(display_list, display_indices, strict=False)):
        ax = plt.subplot(rows, cols, i + 1)

        plt.imshow(img)

        ax.set_title(f"Step: {original_idx + 1}/{len(images)}", fontsize=10)

        ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def display_gif(frames: List[Image.Image], duration: int = 100):
    """
    Generates a GIF animation from a list of PIL.Image frames and displays it directly
    in Jupyter/Colab environment without saving to disk.

    Args:
        frames (List[Image.Image]): List of PIL.Image objects (animation frames).
        duration (int): Duration of each frame in milliseconds.
    """
    if not frames:
        print("Error: Frame list is empty.")
        return

    # Use BytesIO to save GIF in RAM
    buffer = BytesIO()

    # Create GIF and save to buffer
    frames[0].save(
        buffer,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0,  # 0 means infinite loop
    )

    # Reset buffer pointer to the beginning
    buffer.seek(0)

    # Display GIF in notebook using IPython.display
    display(IPImage(data=buffer.read(), format="gif"))


def create_visualization_callback(pipe, storage_list: List[Image.Image]) -> Callable:
    """
    Creates a callback function for capturing intermediate denoising steps for visualization.
    Returns a callback that can be passed to StableDiffusionPipeline.

    Args:
        pipe: StableDiffusionPipeline instance with VAE decoder.
        storage_list (List[Image.Image]): List to store captured images.

    Returns:
        Callable: Callback function compatible with diffusers pipeline.
    """

    def capture_step_visualization(pipeline, step_index: int, timestep: int, callback_kwargs: dict):
        """
        Function called after each denoising step.
        Converts the current latent tensor to an image and saves it.

        Args:
            pipeline: The pipeline instance (passed automatically by diffusers).
            step_index (int): Current denoising step number.
            timestep (int): Current timestep in the diffusion process.
            callback_kwargs (dict): Dictionary containing 'latents' and other tensors.

        Returns:
            dict: Updated callback_kwargs (must return for diffusers compatibility).
        """
        # Extract latents from callback_kwargs
        latents = callback_kwargs["latents"]

        # Take the first element (batch size 1) and detach it from the graph
        latent_to_decode = latents[0].detach()

        # Scale down (decoder requires different scaling)
        latent_to_decode = 1 / pipe.vae.config.scaling_factor * latent_to_decode

        # Pass through VAE decoder
        with torch.no_grad():
            image = pipe.vae.decode(latent_to_decode.unsqueeze(0)).sample

        # Post-processing of the image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).astype("uint8"))

        # Save the image to the storage list
        storage_list.append(image)

        # Must return callback_kwargs for diffusers compatibility
        return callback_kwargs

    return capture_step_visualization


def create_heatmaps_and_overlay(
    activations: Dict[int, Dict[int, np.ndarray]],
    images: Dict[int, Image.Image],
    output_path: Path,
    prompt: str,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.4,
) -> None:
    """
    Create heatmaps from SAE feature activations and overlay on images.

    Args:
        activations: Dict[timestep -> Dict[feature_idx -> activation_array]]
                     where activation_array is 1D array of length seq_len
        images: Dict[timestep -> PIL.Image] - images to overlay on
        output_path: Base path for saving results
        prompt: Prompt text (for organizing output)
        colormap: OpenCV colormap to use (default: COLORMAP_JET)
        alpha: Overlay transparency (0=transparent, 1=opaque)

    Output structure:
        {output_path}/{prompt}/{timestep_XXX}/{feature_XXXXX}/
            - overlay.png: Main result with heatmap overlaid on image
            - heatmap.png: Pure heatmap visualization
            - original.png: Original image without overlay
    """
    print(f"\n{'=' * 80}")
    print("Creating Heatmaps and Overlays")
    print(f"{'=' * 80}")

    # Create output directory structure
    prompt_safe = prompt.replace(" ", "_").replace("/", "_")[:50]
    base_dir = output_path / prompt_safe
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {base_dir}")

    total_saved = 0

    for timestep, step_activations in activations.items():
        print(f"\nProcessing timestep {timestep}...")
        image = images[timestep]
        img_array = np.array(image)
        img_h, img_w = img_array.shape[:2]

        for feature_idx, activation in step_activations.items():
            print(f"  Feature {feature_idx}...", end=" ")

            # 1. Reshape activation to spatial dimensions
            seq_len = len(activation)
            spatial_size = int(np.sqrt(seq_len))

            # Pad if not perfect square
            if spatial_size * spatial_size != seq_len:
                target_len = spatial_size * spatial_size
                if seq_len < target_len:
                    activation = np.pad(
                        activation,
                        (0, target_len - seq_len),
                        mode="constant",
                    )
                else:
                    activation = activation[:target_len]

            heatmap_2d = activation.reshape(spatial_size, spatial_size)
            # 2. Normalize to [0, 1]
            if heatmap_2d.max() > heatmap_2d.min():
                heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min())
            # 3. Resize to image dimensions
            heatmap_resized = cv2.resize(
                heatmap_2d,
                (img_w, img_h),
                # interpolation=cv2.INTER_LINEAR,
            )

            # 4. Apply colormap
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # 5. Overlay on image
            overlaid = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
            overlaid_image = Image.fromarray(overlaid)

            # 6. Save to {output_path}/{prompt}/{timestep}/{feature_nr}/
            feature_dir = base_dir / f"timestep_{timestep:03d}" / f"feature_{feature_idx:05d}"
            feature_dir.mkdir(parents=True, exist_ok=True)

            # Save overlay
            overlay_path = feature_dir / "overlay.png"
            overlaid_image.save(overlay_path)

            # Also save pure heatmap and original image for reference
            heatmap_img = Image.fromarray((heatmap_resized * 255).astype(np.uint8))
            heatmap_img.save(feature_dir / "heatmap.png")
            image.save(feature_dir / "original.png")

            print(f"✓ Saved to {feature_dir.relative_to(output_path)}")
            total_saved += 1

    print(f"\n✓ Saved {total_saved} feature visualizations")
