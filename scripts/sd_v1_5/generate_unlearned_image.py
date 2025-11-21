#!/usr/bin/env python3
"""
Generate images with Stable Diffusion and capture layer representations for unlearned concepts.

Structure: {results_dir}/{model_name}/cached_representations/{layer_name}/
Each dataset contains: object, style, prompt_nr, prompt_text, representation

EXAMPLE:
uv run scripts/sd_v1_5/generate_unlearned_image.py \
    --prompt "A photo of a cat sitting on a table" \
    --preferred_device cuda \
    --guidance_scale 7.5 \
    --steps 50 \
    --seed 42 \
    --output_dir /mnt/evafs/groups/mi2lab/jcwalina/results/test \
    --sae_path /mnt/evafs/groups/mi2lab/mjarosz/results_npy/finetuned_sd_saeuron/sae/unet_up_1_att_1_sae.pt \
    --concept_means_path /mnt/evafs/groups/mi2lab/mjarosz/results_npy/finetuned_sd_saeuron/sae_scores/unet_up_1_att_1_concept_object_cats.npy \
    --influence_factor 1.0 \
    --features_number 25 \
    --epsilon 1e-8 \
"""  # noqa: E501

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusionPipeline  # noqa: E402
from dotenv import load_dotenv
from overcomplete.sae import TopKSAE

import wandb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

# from src.data import load_prompts_from_directory  # noqa: E402
# from src.data.cache import RepresentationCache  # noqa: E402
# from src.models.config import ModelRegistry  # noqa: E402
from src.models.sd_v1_5 import LayerPath  # noqa: E402
from src.models.sd_v1_5.hooks import capture_layer_representations_with_unlearning  # noqa: E402

# from src.utils.model_loader import ModelLoader  # noqa: E402
from src.utils.RepresentationModifier import RepresentationModifier  # noqa: E402


def parse_layer_names(layer_names: List[str]) -> List[LayerPath]:
    """
    Convert string layer names to LayerPath enum values.

    Args:
        layer_names: List of layer name strings

    Returns:
        List of LayerPath enum values
    """
    layers = []
    for name in layer_names:
        try:
            layer = LayerPath[name.upper()]
            layers.append(layer)
        except KeyError:
            print(f"Warning: Unknown layer name '{name}', skipping...")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images with Stable Diffusion + SAE unlearning"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument(
        "--preferred_device",
        type=str,
        default="cuda",
        help="Preferred device (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory (optional, overrides default)",
    )
    parser.add_argument(
        "--sae_path",
        type=str,
        required=True,  # lub default=None jeśli chcesz opcjonalny
        help="Path to SAE weights (.pt)",
    )
    parser.add_argument(
        "--concept_means_path",
        type=str,
        required=True,
        help="Path to concept means (true/false)",
    )
    parser.add_argument(
        "--influence_factor",
        type=float,
        default=1.0,
        help="Influence factor for unlearning",
    )
    parser.add_argument(
        "--features_number",
        type=int,
        default=25,
        help="Number of top features to unlearn",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Small value to avoid division by zero (default: 1e-8)",
    )
    parser.add_argument("--skip_wandb", action="store_true", help="Skip wandb logging")

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        # Initialize wandb
        if not args.skip_wandb:
            wandb.login()
            wandb.init(
                project="sd-control-representation",
                entity="bartoszjezierski28-warsaw-university-of-technology",
                name=f"Image_Generation_SAE_Unlearning {Path(args.sae_path).stem} ",
                config={
                    "sae_path": args.sae_path,
                    "concept_means_path": args.concept_means_path,
                    "epsilon": args.epsilon,
                },
                tags=["sae", "feature_selection"],
                notes="Image generation with Stable Diffusion and SAE unlearning",
            )

        # Get SLURM environment variables
        job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job_id")
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "no_slurm_task_id")

        # Get results directory from environment variable or use default
        # If RESULTS_DIR is set, use it; otherwise use local "results" directory
        results_base_dir = os.environ.get("RESULTS_DIR")
        if results_base_dir:
            results_base_path = Path(results_base_dir)
        else:
            # Default to relative path from project root
            results_base_path = Path(__file__).parent.parent.parent / "results"

        # Set device
        device = "cuda" if args.preferred_device == "cuda" and torch.cuda.is_available() else "cpu"

        print("=" * 50)
        print("Test - Image Generation with Stable Diffusion")
        print("=" * 50)
        print(f"[Job {job_id} | Task {task_id}]")
        print(f"[Prompt: {args.prompt}]")
        print(f"[Preferred Device: {args.preferred_device}]")
        print(f"[Actual Device: {device}]")
        print(f"[Guidance Scale: {args.guidance_scale}]")
        print(f"[Steps: {args.steps}]")
        print(f"[Seed: {args.seed}]")
        print(f"[Results Base Dir: {results_base_path}]")
        print("=" * 50)

        #############################################
        # MODEL
        #############################################
        model_load_start = time.time()
        # loader = ModelLoader(model_enum=ModelRegistry.FINETUNED_SAEURON)
        # pipe = loader.load_model(device=device)
        model_id = "sd-legacy/stable-diffusion-v1-5"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,  # None - to better analyze the behavior of the raw model
        ).to(device)
        model_load_time = time.time() - model_load_start
        print(f"Model loaded in {model_load_time:.2f} seconds")

        #############################################
        # Load SAE → infer config
        #############################################
        sae_path = Path(args.sae_path)
        if not sae_path.exists():
            raise FileNotFoundError(f"SAE not found: {sae_path}")

        # Load state dict
        state_dict = torch.load(sae_path, map_location="cpu")

        # Automatyczne wykrycie i usunięcie prefiksu _orig_mod.
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            print("Detected torch.compile() prefix → removing '_orig_mod.'")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Wnioskowanie wymiarów
        enc_weight = state_dict["encoder.final_block.0.weight"]
        input_shape = enc_weight.shape[1]
        nb_concepts = enc_weight.shape[0]
        print(f"Inferred input_dim={input_shape}, nb_concepts={nb_concepts}")

        # Tworzymy model i ładujemy wagi
        sae = TopKSAE(
            input_shape=input_shape,
            nb_concepts=nb_concepts,
            device=device,
        )
        sae.load_state_dict(state_dict)  # teraz działa idealnie
        sae = sae.to(device)
        sae.eval()
        sae_layer_path = LayerPath.UNET_UP_1_ATT_1  # Example layer path for SAE unlearning
        print("SAE loaded and moved to device")

        #############################################
        # Prepare concept vector
        #############################################
        concept_scores_path = Path(args.concept_means_path)
        if not concept_scores_path.exists():
            raise FileNotFoundError(f"Concept means not found: {concept_scores_path}")

        # file is a disc {'true': tensor([...]), 'false': tensor([...])} saved in format .npy
        concept_means = torch.load(concept_scores_path, map_location="cpu")
        if "mean_true" not in concept_means or "mean_false" not in concept_means:
            raise ValueError("Concept means file must contain 'mean_true' and 'mean_false' keys.")

        modifier = RepresentationModifier(
            sae=sae,
            means_true=concept_means["mean_true"].to(device),
            means_false=concept_means["mean_false"].to(device),
            influence_factor=args.influence_factor,
            features_number=args.features_number,
            epsilon=args.epsilon,
        )
        modifier.attach_to(pipe, sae_layer_path)

        #############################################
        # GENERATION & REPRESENTATION CAPTURING
        #############################################
        layers_to_capture = [
            # Text conditioning
            LayerPath.TEXT_EMBEDDING_FINAL,
            # Critical attention layers
            LayerPath.UNET_MID_ATT,
            LayerPath.UNET_DOWN_2_ATT_0,
            LayerPath.UNET_UP_1_ATT_2,
            # ResNet features for comparison
            LayerPath.UNET_DOWN_1_RES_0,
            LayerPath.UNET_MID_RES_1,
            LayerPath.UNET_UP_0_RES_2,
        ]

        generator = torch.Generator(device).manual_seed(args.seed)
        print("\nGenerating unlearned image and capturing representations...")
        inference_start = time.time()

        # Capture representations and generate image simultaneously
        representations, image = capture_layer_representations_with_unlearning(
            pipe=pipe,
            prompt=args.prompt,
            layer_paths=layers_to_capture,
            modifier=modifier,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )

        inference_time = time.time() - inference_start
        num_repr = len(representations)
        print(
            f"Image generated and {num_repr} representations captured in {inference_time:.2f} seconds"  # noqa: E501
        )

        #############################################
        # SAVE OUTPUTS
        #############################################
        # Determine output directory
        if args.output_dir:
            # Use custom output directory if specified
            output_dir = Path(args.output_dir)
        else:
            # Use RESULTS_DIR/images or default results/images
            output_dir = results_base_path / "images"

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving to: {output_dir}")

        # Create filename with timestamp and task info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_prompt = "".join(c for c in args.prompt[:30] if c.isalnum() or c in (" ", "_")).strip()
        save_prompt = save_prompt.replace(" ", "_")

        image_filename = f"task_{task_id}_{timestamp}_{save_prompt}.png"
        image_path = output_dir / image_filename

        # Save image
        image.save(image_path)
        print(f"\n✓ Image saved to: {image_path}")

        # Save representations
        repr_dir = output_dir / "representations"
        repr_dir.mkdir(parents=True, exist_ok=True)

        repr_base_filename = f"task_{task_id}_{timestamp}_{save_prompt}"
        saved_repr_files = []

        print(f"\nSaving {len(representations)} layer representations...")
        for i, (layer_path, repr_tensor) in enumerate(
            zip(layers_to_capture, representations, strict=True)
        ):
            layer_name = layer_path.name
            repr_filename = f"{repr_base_filename}_{layer_name}.pt"
            repr_file_path = repr_dir / repr_filename

            # Save tensor
            torch.save(repr_tensor, repr_file_path)
            saved_repr_files.append(repr_filename)

            # Print shape info
            shape_str = f"{tuple(repr_tensor.shape)}"
            print(f"  [{i + 1}/{len(representations)}] {layer_name}: {shape_str} → {repr_filename}")

        print(f"\n✓ All representations saved to: {repr_dir}")

        # Save metadata
        metadata_file = output_dir / f"task_{task_id}_{timestamp}_metadata.txt"
        with open(metadata_file, "w") as f:
            f.write(f"Task {task_id}\n")
            f.write(f"Job: {job_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Prompt: {args.prompt}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Guidance Scale: {args.guidance_scale}\n")
            f.write(f"Steps: {args.steps}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Model Load Time: {model_load_time:.2f}s\n")
            f.write(f"Inference Time: {inference_time:.2f}s\n")
            f.write(f"Total Time: {model_load_time + inference_time:.2f}s\n")
            f.write(f"Image: {image_filename}\n")
            f.write(f"\nCaptured Representations ({len(representations)} layers):\n")
            for i, (layer_path, repr_tensor) in enumerate(
                zip(layers_to_capture, representations, strict=True)
            ):
                f.write(f"  {i + 1}. {layer_path.name}: {tuple(repr_tensor.shape)}\n")
                f.write(f"     File: {saved_repr_files[i]}\n")

        print(f"✓ Metadata saved to: {metadata_file}")
        print("\nTiming Summary:")
        print(f"  Model Load: {model_load_time:.2f}s")
        print(f"  Inference: {inference_time:.2f}s")
        print(f"  Total: {model_load_time + inference_time:.2f}s")
        print("\nOutput Summary:")
        print(f"  Image: {image_path}")
        print(f"  Representations: {len(representations)} layers in {repr_dir}")
        print(f"  Metadata: {metadata_file}")
        print("=" * 50)

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
