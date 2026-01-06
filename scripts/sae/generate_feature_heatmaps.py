#!/usr/bin/env python3
"""
Generate heatmaps of SAE feature activations overlaid on images.

This script:
1. Generates images while capturing representations at specific timesteps
2. Passes representations through SAE to get feature activations
3. Creates heatmaps from activations and overlays them on images
4. Saves results organized by prompt/timestep/feature

Example usage:
uv run scripts/sae/generate_feature_heatmaps.py \
    --sae_path /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp16_topk32_lr5em5_ep2_bs4096/model.pt \
    --feature_list 1 2 3 \
    --prompt "naked woman" \
    --layer UNET_UP_1_ATT_1 \
    --timesteps 2 10 20 30 40 50 \
    --output_path /mnt/evafs/groups/mi2lab/bjezierski/results/sd_v1_5/analysis/heatmaps \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --seed 42 \
    --alpha 0.4

Or:
uv run scripts/sae/generate_feature_heatmaps.py \
    --sae_path /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp16_topk32_lr5em5_ep2_bs4096/model.pt \
    --feature_selection_results "/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp16_topk32_lr5em5_ep2_bs4096/feature_sums/unet_up_1_att_1_concept_object_exposed armpits_job1_851850c4.pt" \
    --top_k_features 5 \
    --prompt "naked woman with exposed armpits" \
    --layer UNET_UP_1_ATT_1 \
    --timesteps 2 10 20 30 40 50 \
    --output_path /mnt/evafs/groups/mi2lab/bjezierski/results/sd_v1_5/analysis/heatmaps \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --seed 42 \
    --alpha 0.4

Or:
uv run scripts/sae/generate_feature_heatmaps.py \
    --sae_path /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp16_topk32_lr5em5_ep2_bs4096/model.pt \
    --top_k_features 5 \
    --prompt "naked woman with exposed butt" \
    --layer UNET_UP_1_ATT_1 \
    --timesteps 49 \
    --output_path /mnt/evafs/groups/mi2lab/bjezierski/results/sd_v1_5/analysis/heatmaps \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --seed 42 \
    --alpha 0.4 \
    --select_active_features
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from overcomplete.sae import TopKSAE
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.config import ModelRegistry  # noqa: E402
from src.models.sd_v1_5.hooks import capture_layer_representations  # noqa: E402
from src.models.sd_v1_5.layers import LayerPath  # noqa: E402
from src.utils.model_loader import ModelLoader  # noqa: E402
from src.utils.visualization import create_heatmaps_and_overlay  # noqa: E402


def generate_and_collect_activations(
    timesteps: List[int],
    feature_indices: Dict[int, List[int]] | List[int] | None,
    sae_path: Path,
    prompt: str,
    layer: LayerPath,
    pipe,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    device: str = "cuda",
    select_active_features: bool = False,
    top_k_features: int = 5,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, Image.Image]]:
    """
    Generate image and collect SAE activations for specific features and timesteps.

    Args:
        timesteps: List of timestep indices to capture (e.g., [0, 10, 20, 30,,
                        or None if select_active_features=True
        sae_path: Path to saved SAE model.pt
        prompt: Text prompt for image generation
        layer: LayerPath enum specifying which layer to capture
        pipe: Stable Diffusion pipeline (from model_loader)
        num_inference_steps: Number of diffusion steps
        guidance_scale: CFG scale
        seed: Random seed
        device: Device to run on
        select_active_features: If True, auto-select features that activate during generation
        top_k_features: Number of top active features to select (if select_active_features=True)e
        seed: Random seed
        device: Device to run on

    Returns:
        Tuple of (activations, images) where:
        - activations: Dict[timestep -> Dict[feature_idx -> activation_array]]
        - images: Dict[timestep -> PIL.Image] (predicted x0 images at each timestep)
    """
    print(f"\n{'=' * 80}")
    print("Generate and Collect SAE Activations")
    print(f"{'=' * 80}")
    print(f"Prompt: '{prompt}'")
    print(f"Layer: {layer.name}")
    print(f"Timesteps to capture: {timesteps}")
    if select_active_features:
        print(f"Mode: AUTO-SELECT ACTIVE FEATURES (development mode)")
        print(f"Will select top {top_k_features} features that activate at each timestep")
    elif isinstance(feature_indices, dict):
        print(f"Features to extract: per-timestep (dict with {len(feature_indices)} timesteps)")
    else:
        print(f"Features to extract: {feature_indices}")
        print(f"Features to extract: {feature_indices}")

    # Load SAE model
    print(f"\nLoading SAE from {sae_path}...")
    state_dict = torch.load(sae_path, map_location="cpu")

    # Remove torch.compile() prefix if present
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Infer SAE dimensions
    enc_weight = state_dict["encoder.final_block.0.weight"]
    input_shape = enc_weight.shape[1]
    nb_concepts = enc_weight.shape[0]
    top_k = state_dict.get("top_k", 32)  # Try to infer from state_dict

    # Create SAE
    sae = TopKSAE(
        input_shape=input_shape,
        nb_concepts=nb_concepts,
        top_k=top_k,
        device=device,
    )
    sae.load_state_dict(state_dict)
    sae = sae.to(device)
    sae.eval()
    print(f"✓ SAE loaded: {nb_concepts} concepts, input_dim={input_shape}")

    # Generate image and capture representations using existing function
    print(f"\nGenerating image with {num_inference_steps} steps...")
    generator = torch.Generator(device).manual_seed(seed)

    representations, final_image, latents_tensor = capture_layer_representations(
        pipe=pipe,
        prompt=prompt,
        layer_paths=[layer],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        capture_latents=True,  # Enable latent capture
    )

    # representations[0] has shape [num_timesteps, 1, seq_len, feature_dim]
    all_reps = representations[0]  # Representations for SAE
    all_latents = latents_tensor  # [num_timesteps, batch, 4, 64, 64]
    total_timesteps = all_reps.shape[0]

    print(f"✓ Generated image, captured {total_timesteps} timesteps")
    print(f"  Representation shape: {all_reps.shape}")
    print(f"  Latent shape: {all_latents.shape}")

    # Process representations through SAE to get activations
    print("\nExtracting SAE activations for specified features...")
    activations = {}  # timestep -> {feature_idx -> activation_array}
    images = {}  # timestep -> PIL.Image

    # Process each requested timestep
    for step_idx in timesteps:
        if step_idx >= total_timesteps:
            print(f"  Warning: Timestep {step_idx} >= {total_timesteps}, skipping")
            continue

        print(f"  Processing timestep {step_idx}...", end=" ")

        # Get representation: [1, seq_len, feature_dim]
        rep = all_reps[step_idx]

        # Reshape for SAE: [1, seq_len, features] -> [seq_len, features]
        if rep.ndim == 3:
            rep = rep.squeeze(0)

        # Forward through SAE
        with torch.no_grad():
            rep_device = rep.to(device).float()
            _, codes, _ = sae(rep_device)  # codes: [seq_len, num_concepts]
            codes_cpu = codes.cpu().numpy()

        # Extract activations for requested features
        step_activations = {}

        # Get features for this timestep
        if select_active_features:
            # Development mode: select top-K most active features at this timestep
            # Sum activations across all spatial positions
            activation_sums = np.abs(codes_cpu).sum(axis=0)  # [num_concepts]
            # Get indices of top-K active features
            features_for_step = np.argsort(activation_sums)[-top_k_features:][::-1].tolist()
            print(f"    Auto-selected active features: {features_for_step}")
        elif isinstance(feature_indices, dict):
            features_for_step = feature_indices.get(step_idx, [])
        else:
            features_for_step = feature_indices

        for feat_idx in features_for_step:
            step_activations[feat_idx] = codes_cpu[:, feat_idx]  # [seq_len]

        activations[step_idx] = step_activations

        # Predict x0 image at this timestep using captured latent
        # captured_latents has shape [num_timesteps, batch, 4, 64, 64]
        latent = all_latents[step_idx]  # [batch, 4, 64, 64]
        x0_image = predict_x0_from_latent(latent, pipe, device)
        images[step_idx] = x0_image

        print(f"✓ {len(step_activations)} features extracted")

    print(f"\n✓ Collected activations for {len(activations)} timesteps")

    return activations, images


def predict_x0_from_latent(latent: torch.Tensor, pipe, device: str) -> Image.Image:
    """Predict fully denoised image from latent."""
    with torch.no_grad():
        latent = latent.to(device)
        scaled_latent = latent / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(scaled_latent).sample

        image = (decoded / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).astype("uint8"))

    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SAE feature heatmaps overlaid on images",
    )

    parser.add_argument(
        "--sae_path",
        type=str,
        required=True,
        help="Path to SAE model.pt file",
    )
    parser.add_argument(
        "--feature_selection_results",
        type=str,
        default=None,
        help="Path to feature_selection results (.pt file with feature scores)",
    )
    parser.add_argument(
        "--feature_list",
        type=int,
        nargs="+",
        default=None,
        help="List of specific feature indices to visualize (e.g., 42 100 256)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for image generation",
    )
    parser.add_argument(
        "--layer",
        type=str,
        required=True,
        help="Layer name (e.g., UNET_UP_1_ATT_1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="FINETUNED_SAEURON",
        help="Model to use from ModelRegistry",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[0, 10, 20, 30, 40],
        help="Timestep indices to visualize",
    )
    parser.add_argument(
        "--top_k_features",
        type=int,
        default=5,
        help="Number of top features to visualize (used with --feature_selection_results)",
    )
    parser.add_argument(
        "--select_active_features",
        action="store_true",
        help="Development mode: select features that actually activated during generation (ignores feature_selection_results)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Base output directory",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Heatmap overlay transparency (0-1)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Parse layer
    try:
        layer = LayerPath[args.layer]
    except KeyError:
        print(f"Error: Invalid layer '{args.layer}'")
        print(f"Available layers: {[l.name for l in LayerPath]}")
        return 1

    # Load model using ModelLoader
    print(f"\nLoading model: {args.model}")
    try:
        model_enum = ModelRegistry[args.model]
    except KeyError:
        print(f"Error: Invalid model '{args.model}'")
        print(f"Available models: {[m.name for m in ModelRegistry]}")
        return 1

    model_loader = ModelLoader(model_enum)
    pipe = model_loader.load_model(device=device)
    print("✓ Model loaded")

    # Determine which features to visualize
    if args.select_active_features:
        # Development mode: features will be auto-selected during generation
        print(f"\n✓ Using ACTIVE FEATURE SELECTION mode (development)")
        print(
            f"   Will automatically select top {args.top_k_features} active features per timestep"
        )
        feature_indices = None  # Will be determined during generation
    elif args.feature_list is not None:
        # Use manually specified feature list for all timesteps
        feature_indices = args.feature_list
        print(f"\n✓ Using manually specified features: {feature_indices}")
    elif args.feature_selection_results is not None:
        # Load feature selection results to get top features per timestep
        print(f"\nLoading feature selection results from {args.feature_selection_results}")
        feature_data = torch.load(args.feature_selection_results)

        # Debug: Check what keys are actually in the file
        print(f"Available keys in feature_data: {list(feature_data.keys())}")

        # Handle different possible formats
        if "sums_true_per_timestep" in feature_data:
            # New format from feature_selection.py with explicit true/false splits
            sums_true = feature_data["sums_true_per_timestep"]
            sums_false = feature_data["sums_false_per_timestep"]
            counts_true = feature_data["counts_true_per_timestep"]
            counts_false = feature_data["counts_false_per_timestep"]
        elif "sums_per_timestep" in feature_data:
            # Alternative format - single concept data (from RepresentationModifier style)
            # This represents activations for concept=True
            # We'll compute importance based on activation magnitude alone
            print(f"Note: Using single-concept format (no true/false split)")
            print(f"Computing importance based on activation magnitude per timestep")

            sums_concept = feature_data["sums_per_timestep"]
            counts_concept = feature_data["counts_per_timestep"]

            # Calculate top features per timestep based on mean activation magnitude
            feature_indices = {}
            epsilon = 1e-10

            for timestep in args.timesteps:
                if timestep not in sums_concept:
                    print(
                        f"  Warning: Timestep {timestep} not in feature selection results, skipping"
                    )
                    continue

                # Compute mean activation for this timestep
                mean_activation = sums_concept[timestep] / max(counts_concept[timestep], 1)

                # Normalize to probabilities
                prob_activation = mean_activation / (mean_activation.sum() + epsilon)

                # Get top-K features by activation probability
                top_features = torch.topk(prob_activation, k=args.top_k_features).indices.tolist()
                feature_indices[timestep] = top_features
                print(f"  Timestep {timestep}: top {args.top_k_features} features = {top_features}")

            print(
                f"✓ Selected top {args.top_k_features} features per timestep for {len(feature_indices)} timesteps"
            )
        else:
            print(f"Error: Unrecognized feature selection file format")
            print(f"Expected keys: 'sums_true_per_timestep' or 'sums_per_timestep'")
            print(f"Found keys: {list(feature_data.keys())}")
            return 1

        # Only run the true/false comparison logic if we have that data
        if "sums_true_per_timestep" in feature_data:
            # Calculate top features per timestep using probability difference
            feature_indices = {}
            epsilon = 1e-10

            for timestep in args.timesteps:
                # Check if timestep exists in the data
                if timestep not in sums_true:
                    print(
                        f"  Warning: Timestep {timestep} not in feature selection results, skipping"
                    )
                    continue

                # Compute importance scores (probability difference) for this timestep
                # Extract data for this specific timestep
                mean_true = sums_true[timestep] / counts_true[timestep]
                mean_false = sums_false[timestep] / counts_false[timestep]

                # Normalize to probabilities
                prob_true = mean_true / (mean_true.sum() + epsilon)
                prob_false = mean_false / (mean_false.sum() + epsilon)

                # Importance is the difference in probabilities
    elif not args.select_active_features:
        # Only error if not in active feature selection mode
        print(
            "Error: Must specify either --feature_list, --feature_selection_results, or --select_active_features"
        )
        return 1

    # Generate and collect activations
    # Generate and collect activations
    activations, images = generate_and_collect_activations(
        timesteps=args.timesteps,
        feature_indices=feature_indices,
        sae_path=Path(args.sae_path),
        prompt=args.prompt,
        layer=layer,
        pipe=pipe,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=device,
        select_active_features=args.select_active_features,
        top_k_features=args.top_k_features,
    )

    # Create heatmaps and overlay
    create_heatmaps_and_overlay(
        activations=activations,
        images=images,
        output_path=Path(args.output_path),
        prompt=args.prompt,
        alpha=args.alpha,
    )

    print(f"\n{'=' * 80}")
    print("✓ Complete! Results saved to:")
    print(f"  {Path(args.output_path) / args.prompt.replace(' ', '_')[:50]}")
    print(f"{'=' * 80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
