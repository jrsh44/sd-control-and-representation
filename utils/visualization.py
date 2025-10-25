import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Callable
import torch

from io import BytesIO
from IPython.display import display, Image as IPImage


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
    if not isinstance(images, list) or not all(
        isinstance(img, Image.Image) for img in images
    ):
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
    for i, (img, original_idx) in enumerate(zip(display_list, display_indices)):
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

    def capture_step_visualization(
        pipeline, step_index: int, timestep: int, callback_kwargs: dict
    ):
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
