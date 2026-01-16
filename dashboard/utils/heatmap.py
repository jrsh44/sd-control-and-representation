"""
Feature Heatmap Utilities Module

Generates SAE feature activation heatmaps overlaid on generated images.
Used to visualize which spatial regions activate specific SAE features
during the Stable Diffusion denoising process.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from core.state import DashboardState


def collect_activations_from_representations(
    representations: torch.Tensor,
    sae,
    timesteps: List[int],
    top_k_features: int = 5,
    device: str = "cuda",
    feature_scores: Optional[torch.Tensor] = None,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Process representations through SAE to get feature activations.
    Auto-selects the top-K most active features at each timestep,
    or uses provided feature_scores for concept-based selection.

    Args:
        representations: Tensor of shape [num_timesteps, batch, seq_len, feature_dim]
        sae: TopKSAE model
        timesteps: List of timestep indices to process
        top_k_features: Number of top features to select per timestep
        device: Device to run SAE on
        feature_scores: Optional pre-computed feature scores from RepresentationModifier
                       Shape: [num_timesteps, num_features] for per-timestep or
                              [num_features] for global

    Returns:
        Dict[timestep -> Dict[feature_idx -> activation_array]]
    """
    total_timesteps = representations.shape[0]
    activations = {}

    for step_idx in timesteps:
        if step_idx >= total_timesteps:
            continue

        # Get representation: [batch, seq_len, feature_dim]
        rep = representations[step_idx]

        # Reshape for SAE: [batch, seq_len, features] -> [seq_len, features]
        if rep.ndim == 3:
            rep = rep.squeeze(0)

        # Forward through SAE
        with torch.no_grad():
            rep_device = rep.to(device).float()
            _, codes, _ = sae(rep_device)  # codes: [seq_len, num_concepts]
            codes_cpu = codes.cpu().numpy()

        # Select features for this timestep
        if feature_scores is not None:
            # Use provided feature scores from RepresentationModifier
            if feature_scores.ndim == 2:  # Per-timestep scores
                if step_idx < feature_scores.shape[0]:
                    step_scores = feature_scores[step_idx]
                    top_features = torch.topk(step_scores, k=top_k_features).indices.tolist()
                else:
                    # Fallback to activation-based if timestep out of range
                    activation_sums = np.abs(codes_cpu).sum(axis=0)
                    top_features = np.argsort(activation_sums)[-top_k_features:][::-1].tolist()
            else:  # Global scores
                top_features = torch.topk(feature_scores, k=top_k_features).indices.tolist()
        else:
            # Auto-select top-K most active features
            activation_sums = np.abs(codes_cpu).sum(axis=0)  # [num_concepts]
            top_features = np.argsort(activation_sums)[-top_k_features:][::-1].tolist()

        step_activations = {}
        for feat_idx in top_features:
            step_activations[feat_idx] = codes_cpu[:, feat_idx]  # [seq_len]

        activations[step_idx] = step_activations

    return activations


def decode_latent_to_image(latent: torch.Tensor, pipe, device: str) -> Image.Image:
    """
    Decode a latent tensor to an RGB image using the VAE decoder.

    Args:
        latent: Latent tensor of shape [1, 4, 64, 64] or [4, 64, 64]
        pipe: Stable Diffusion pipeline with VAE
        device: Device to run decoding on

    Returns:
        PIL Image decoded from latent
    """
    with torch.no_grad():
        latent = latent.to(device)
        # Ensure batch dimension
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)

        # Decode using VAE
        scaled_latent = latent / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(scaled_latent).sample

        # Convert to image
        image = (decoded / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).astype("uint8"))

    return image


def create_heatmap_overlay(
    activation: np.ndarray,
    image: Image.Image,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.4,
) -> Image.Image:
    """
    Create a heatmap from activation and overlay on image.

    Args:
        activation: 1D array of activations [seq_len]
        image: PIL Image to overlay on
        colormap: OpenCV colormap (default: JET)
        alpha: Overlay transparency (0=transparent, 1=opaque)

    Returns:
        PIL Image with heatmap overlay
    """
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]

    # Reshape activation to spatial dimensions
    seq_len = len(activation)
    spatial_size = int(np.sqrt(seq_len))

    # Pad if not perfect square
    if spatial_size * spatial_size != seq_len:
        target_len = spatial_size * spatial_size
        if seq_len < target_len:
            activation = np.pad(activation, (0, target_len - seq_len), mode="constant")
        else:
            activation = activation[:target_len]

    heatmap_2d = activation.reshape(spatial_size, spatial_size)

    # Normalize to [0, 1]
    if heatmap_2d.max() > heatmap_2d.min():
        heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min())

    # Resize to image dimensions
    heatmap_resized = cv2.resize(heatmap_2d, (img_w, img_h))

    # Apply colormap
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay on image
    overlaid = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)

    return Image.fromarray(overlaid)


def generate_heatmap_gallery(
    activations: Dict[int, Dict[int, np.ndarray]],
    image: Image.Image = None,
    alpha: float = 0.4,
    max_features: int = 5,
    intermediate_images: Optional[Dict[int, Image.Image]] = None,
    nudenet_detector=None,
    apply_censorship: bool = False,
    state: "DashboardState | None" = None,
    concept_name: Optional[str] = None,
) -> List[Tuple[Image.Image, str]]:
    """
    Generate a gallery of heatmap overlays for multiple features/timesteps.

    Args:
        activations: Dict[timestep -> Dict[feature_idx -> activation_array]]
        image: Base image to overlay heatmaps on (fallback if intermediate_images not provided)
        alpha: Overlay transparency
        max_features: Maximum number of features to show per timestep
        intermediate_images: Dict mapping timestep to decoded intermediate image
        nudenet_detector: NudeNet detector for censorship
        apply_censorship: Whether to apply NudeNet censorship to heatmap images
        state: Optional dashboard state for logging
        concept_name: Optional concept name to include in captions

    Returns:
        List of (image, caption) tuples for gallery display
    """
    gallery = []

    # Detect nudity coordinates once from the final image
    nudity_detections = []
    if apply_censorship and nudenet_detector is not None and image is not None:
        from utils.detection import detect_nudity_coordinates

        nudity_detections = detect_nudity_coordinates(image, state, nudenet_detector)
        if nudity_detections:
            if state:
                state.log(
                    f"Detected {len(nudity_detections)} unsafe region(s) in final image, "
                    f"will apply to all heatmaps",
                    "info",
                )
        else:
            if state:
                state.log("No unsafe content detected in final image", "info")

    for timestep in sorted(activations.keys()):
        step_activations = activations[timestep]
        feature_count = 0

        # Use intermediate image for this timestep if available, otherwise fallback to final image
        base_image = intermediate_images.get(timestep, image) if intermediate_images else image

        if base_image is None:
            if state:
                state.log(f"No base image available for timestep {timestep}, skipping", "warning")
            continue

        # Apply censorship using coordinates from final image
        display_image = base_image
        if apply_censorship and nudity_detections:
            from utils.detection import apply_censorship_boxes

            # Scale coordinates if image size differs
            final_w, final_h = image.size
            base_w, base_h = base_image.size
            scale_x = base_w / final_w
            scale_y = base_h / final_h

            # Create scaled detections for this image
            scaled_detections = []
            for det in nudity_detections:
                box = det["box"]
                x, y, w, h = box
                scaled_box = [
                    int(x * scale_x),
                    int(y * scale_y),
                    int(w * scale_x),
                    int(h * scale_y),
                ]
                scaled_detections.append({**det, "box": scaled_box})

            display_image = apply_censorship_boxes(base_image, scaled_detections)
            if state and timestep == sorted(activations.keys())[0]:
                # Log only once for first timestep
                state.log(
                    f"Applied censorship boxes from final image to all {len(activations)} timesteps",
                    "info",
                )

        for feature_idx, activation in step_activations.items():
            if feature_count >= max_features:
                break

            overlay = create_heatmap_overlay(activation, display_image, alpha=alpha)
            caption = f"T{timestep} | Feature {feature_idx}"
            if concept_name:
                caption = f"{concept_name} | {caption}"
            gallery.append((overlay, caption))
            feature_count += 1

    return gallery


def get_activation_score_html(activation: float) -> str:
    """Get formatted activation score HTML for table cell, matching NudeNet style.

    Args:
        activation: Activation strength value

    Returns:
        HTML formatted score string with color indicator
    """
    # Normalize activation to determine color (higher = more significant)
    abs_val = abs(activation)
    if abs_val >= 2.0:
        css_class = "score-high"
        emoji = "🔴"
    elif abs_val >= 1.0:
        css_class = "score-medium"
        emoji = "🟠"
    elif abs_val >= 0.5:
        css_class = "score-low"
        emoji = "🟡"
    else:
        css_class = "score-safe"
        emoji = "🟢"

    return f'<span class="score-value {css_class}">{emoji} {abs_val:.2f}</span>'


def format_heatmap_info(
    activations: Dict[int, Dict[int, np.ndarray]] | Dict[str, Dict[int, Dict[int, np.ndarray]]],
    multi_concept: bool = False,
) -> str:
    """
    Format heatmap information as HTML for display, matching NudeNet table style.

    Args:
        activations: Dict[timestep -> Dict[feature_idx -> activation_array]]
                    or Dict[concept_name -> Dict[timestep -> Dict[feature_idx -> activation_array]]]
        multi_concept: Whether activations contains multiple concepts

    Returns:
        HTML string with heatmap summary
    """
    if not activations:
        return ""

    rows = []

    if multi_concept:
        # Multiple concepts: group by concept
        for concept_name, concept_activations in activations.items():
            rows.append(f"""
                <tr style="background: rgba(94, 129, 172, 0.15);">
                    <td colspan="3">📊 <strong>{concept_name}</strong></td>
                </tr>
            """)
            for timestep in sorted(concept_activations.keys()):
                features = list(concept_activations[timestep].keys())
                for feat_idx in features:
                    act = concept_activations[timestep][feat_idx]
                    max_act = np.abs(act).max()
                    score_html = get_activation_score_html(max_act)
                    rows.append(f"""
                        <tr>
                            <td>Step {timestep}</td>
                            <td>Feature {feat_idx}</td>
                            <td>{score_html}</td>
                        </tr>
                    """)
    else:
        # Single concept or no grouping - show each feature per timestep
        for timestep in sorted(activations.keys()):
            features = list(activations[timestep].keys())
            for feat_idx in features:
                act = activations[timestep][feat_idx]
                max_act = np.abs(act).max()
                score_html = get_activation_score_html(max_act)
                rows.append(f"""
                    <tr>
                        <td>Step {timestep}</td>
                        <td>Feature {feat_idx}</td>
                        <td>{score_html}</td>
                    </tr>
                """)

    return f"""
<div class="analysis-container heatmap-container">
    <div class="analysis-header">
        <h4>SAE Feature Activation Analysis</h4>
        <p class="analysis-description">
            <strong>SAE Features</strong> represent learned concepts in the model's internal representations.
            Each feature's activation strength indicates how strongly that concept is present.
            Higher values (🔴 🟠) indicate stronger activation of the feature.
        </p>
    </div>

    <div class="analysis-content">
        <table class="detection-table">
            <thead>
                <tr>
                    <th>Timestep</th>
                    <th>Feature</th>
                    <th>Activation</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
</div>
"""


def generate_heatmaps_for_dashboard(
    pipe,
    sae,
    prompt: str,
    layer_path,
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    timesteps: List[int] = None,
    top_k_features: int = 5,
    alpha: float = 0.4,
    device: str = "cuda",
    state: "DashboardState | None" = None,
) -> Tuple[List[Tuple[Image.Image, str]], str, Image.Image]:
    """
    Generate feature heatmaps for dashboard display.

    This is a streamlined version that:
    1. Generates an image while capturing representations
    2. Passes representations through SAE to get feature activations
    3. Creates heatmap overlays for top-K active features

    Args:
        pipe: Stable Diffusion pipeline
        sae: TopKSAE model
        prompt: Text prompt
        layer_path: LayerPath enum for capture
        seed: Random seed
        num_inference_steps: Number of diffusion steps
        guidance_scale: CFG scale
        timesteps: Timesteps to capture (default: [10, 25, 40])
        top_k_features: Number of top features per timestep
        alpha: Heatmap overlay transparency
        device: Device to run on
        state: Optional dashboard state for logging

    Returns:
        Tuple of (gallery, info_html, final_image)
    """
    from src.models.sd_v1_5.hooks import capture_layer_representations

    if timesteps is None:
        timesteps = [10, 25, 40]  # Early, mid, late stages

    if state:
        state.log(f"Capturing representations at timesteps: {timesteps}", "info")

    # Generate image and capture representations
    generator = torch.Generator(device=device).manual_seed(seed)

    representations, final_image, latents = capture_layer_representations(
        pipe=pipe,
        prompt=prompt,
        layer_paths=[layer_path],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        capture_latents=True,
    )

    if representations[0] is None:
        if state:
            state.log("Failed to capture representations", "error")
        return [], "", final_image

    # Collect activations from representations
    activations = collect_activations_from_representations(
        representations=representations[0],
        sae=sae,
        timesteps=timesteps,
        top_k_features=top_k_features,
        device=device,
    )

    if state:
        total_features = sum(len(v) for v in activations.values())
        state.log(f"Extracted {total_features} feature activations", "success")

    # Decode intermediate images from latents for specific timesteps
    intermediate_images = {}
    if latents is not None:
        for timestep in timesteps:
            if timestep < latents.shape[0]:
                latent = latents[timestep]
                intermediate_img = decode_latent_to_image(latent, pipe, device)
                intermediate_images[timestep] = intermediate_img
                if state:
                    state.log(f"Decoded intermediate image at timestep {timestep}", "info")

    # Generate heatmap gallery with intermediate images
    gallery = generate_heatmap_gallery(
        activations=activations,
        image=final_image,
        alpha=alpha,
        max_features=top_k_features,
        intermediate_images=intermediate_images,
    )

    # Format info HTML
    info_html = format_heatmap_info(activations)

    return gallery, info_html, final_image
