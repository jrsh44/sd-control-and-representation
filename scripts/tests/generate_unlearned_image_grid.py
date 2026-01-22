#!/usr/bin/env python3
"""
EXAMPLE:
uv run scripts/tests/generate_unlearned_image_grid.py \
    --prompt "a girl with exposed anus on a bed" \
    --preferred_device cpu \
    --guidance_scale 4 \
    --steps 50 \
    --seed 42 \
    --output_dir /mnt/evafs/groups/mi2lab/jcwalina/results/test \
    --sae_dir_path /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096 \
    --concept_sums_path /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/feature_merged/merged_feature_sums.pt \
    --epsilon 1e-8 \
    --ignore_modification false \
    --layers UNET_UP_1_ATT_2 UNET_DOWN_1_RES_0 \
    --skip_wandb \
    --concept_name "breast" \
    --parameter_pairs 5 2 \
    --parameter_pairs 5 5 \
    --parameter_pairs 5 10 \
    --parameter_pairs 8 2 \
    --parameter_pairs 8 5 \
    --parameter_pairs 8 10 \
    --parameter_pairs 2 20

"""  # noqa: E501

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

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

from src.models.sd_v1_5.hooks import capture_layer_representations_with_unlearning  # noqa: E402
from src.models.sd_v1_5.layers import LayerPath  # noqa: E402
from src.utils.RepresentationModifier import RepresentationModifier  # noqa: E402
from src.utils.script_functions import parse_layer_names  # noqa: E402


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
        "--sae_dir_path",
        type=str,
        required=True,
        help="Path to SAE weights (.pt)",
    )
    parser.add_argument(
        "--concept_sums_path",
        type=str,
        required=True,
        help="Path to concept sums (true/false)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Small value to avoid division by zero (default: 1e-8)",
    )
    parser.add_argument(
        "--ignore_modification",
        type=str,
        default="true",
        help="Type of generation: 'true' (no modification), 'false' (with modification)",  # noqa: E501
    )
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        required=True,
        help="List of layer names to capture",
    )
    parser.add_argument("--skip_wandb", action="store_true", help="Skip wandb logging")
    parser.add_argument(
        "--concept_name",
        type=str,
        required=True,
        help="Name of the concept to unlearn",
    )
    parser.add_argument(
        "--parameter_pairs",
        action="append",
        nargs=2,
        type=int,
        metavar=("FEATURES_NUMBER", "INFLUENCE_FACTOR"),
        help="Parameter pairs (features_number influence_factor). Can be specified multiple times. "
        "Example: --parameter_pairs 2 10 --parameter_pairs 5 25",
    )
    parser.add_argument(
        "--per_timestep",
        action="store_true",
        help="Use per-timestep mode for unlearning",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate parameter_pairs
    if not args.parameter_pairs:
        print("ERROR: At least one parameter pair must be specified", file=sys.stderr)
        return 1

    try:
        # Initialize wandb
        if not args.skip_wandb:
            wandb.login()
            wandb.init(
                project="sd-control-representation",
                entity="bartoszjezierski28-warsaw-university-of-technology",
                name=f"Image_Generation_SAE_Unlearning_Grid_{Path(args.sae_dir_path).stem}",
                config={
                    "sae_dir_path": args.sae_dir_path,
                    "concept_sums_path": args.concept_sums_path,
                    "epsilon": args.epsilon,
                    "concept_name": args.concept_name,
                    "parameter_pairs": args.parameter_pairs,
                },
                tags=["image_grid", "sae", "unlearning"],
                notes="Image generation grid with Stable Diffusion and SAE unlearning",
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

        # --------------------------------------------------------------------------
        # 1. MODEL
        # --------------------------------------------------------------------------
        model_load_start = time.time()
        model_id = "sd-legacy/stable-diffusion-v1-5"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
        model_load_time = time.time() - model_load_start
        print(f"Model loaded in {model_load_time:.2f} seconds")

        # --------------------------------------------------------------------------
        # 2. Load SAE → infer config
        # --------------------------------------------------------------------------
        sae_dir_path = Path(args.sae_dir_path)
        if not sae_dir_path.exists():
            raise FileNotFoundError(f"SAE directory not found: {sae_dir_path}")

        # Find the first model.pt or checkpoint.pt file in the sae_dir_path
        pt_files = list(sae_dir_path.glob("model.pt")) + list(sae_dir_path.glob("checkpoint.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No model.pt or checkpoint.pt file found in {sae_dir_path}")

        sae_path = pt_files[0]
        print(f"Using SAE weights file: {sae_path}")

        # extract topk and batch_size from sae_path e.g. wds_nudity/unet_up_1_att_1/exp8_topk16_lr4em4_ep5_bs4096  # noqa: E501
        match = re.search(r"topk(\d+)_.*_bs(\d+)", str(sae_path))
        if match:
            extracted_topk = int(match.group(1))
            print(f"Extracted top_k={extracted_topk} from SAE path")
        else:
            print("Warning: Could not extract top_k and batch_size from SAE path")
            extracted_topk = 32
            print(f"Using default top_k={extracted_topk}")

        # Load state dict
        sae_dict = torch.load(sae_path, map_location="cpu")

        state_dict = sae_dict.get("model_state_dict", sae_dict)

        # print keys in state_dict (for tests)
        print(f"SAE state_dict keys: {list(state_dict.keys())}")

        # Auto-detect and remove _orig_mod prefix from torch.compile()
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            print("Detected torch.compile() prefix → removing '_orig_mod.'")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Infer dimensions from weights
        enc_weight = state_dict["encoder.final_block.0.weight"]
        input_shape = enc_weight.shape[1]
        nb_concepts = enc_weight.shape[0]
        print(f"Inferred input_dim={input_shape}, nb_concepts={nb_concepts}")

        # Create model and load weights
        sae = TopKSAE(
            input_shape=input_shape,
            nb_concepts=nb_concepts,
            top_k=extracted_topk,
            device=device,
        )
        sae.load_state_dict(state_dict)
        sae = sae.to(device)
        sae.eval()
        print("✓ SAE loaded and moved to device")

        sae_layer_path = LayerPath.UNET_UP_1_ATT_1  # hardcoded for test

        # --------------------------------------------------------------------------
        # 3. Prepare RepresentationModifier
        # --------------------------------------------------------------------------
        concept_scores_path = Path(args.concept_sums_path)
        if not concept_scores_path.exists():
            raise FileNotFoundError(f"Concept means not found: {concept_scores_path}")

        concept_sums = torch.load(concept_scores_path, map_location=device)

        # --------------------------------------------------------------------------
        # 4. GENERATION & REPRESENTATION CAPTURING
        # --------------------------------------------------------------------------
        layers_to_capture = parse_layer_names(args.layers)
        if not layers_to_capture:
            print("ERROR: No valid layers specified")
            return 1

        print(
            f"\nGenerating {len(args.parameter_pairs)} images for concept '{args.concept_name}'..."
        )
        print(f"Parameter pairs: {args.parameter_pairs}")

        # Find max features_number to set max_concepts_number appropriately
        max_features = max(fn for fn, _ in args.parameter_pairs)

        # Create modifier once and attach to pipeline
        modifier = RepresentationModifier(
            sae=sae,
            stats_dict=concept_sums,
            epsilon=args.epsilon,
            ignore_modification=args.ignore_modification,
            device=device,
            max_concepts_number=max_features,
        )
        modifier.attach_to(pipe, sae_layer_path)

        # Add the concept once with initial parameters
        initial_features, initial_influence = args.parameter_pairs[0]
        modifier.add_concept_to_unlearn(
            concept_name=args.concept_name,
            influence_factor=initial_influence,
            features_number=initial_features,
            per_timestep=args.per_timestep,
        )
        print(f"Concept '{args.concept_name}' added to modifier (per_timestep={args.per_timestep})")

        # Loop through each parameter pair and generate images
        for pair_idx, (features_number, influence_factor) in enumerate(args.parameter_pairs, 1):
            print(f"\n{'=' * 50}")
            print(
                f"[{pair_idx}/{len(args.parameter_pairs)}] features={features_number}, influence={influence_factor}"
            )
            print(f"{'=' * 50}")

            # Update parameters for this configuration
            modifier.set_number_of_features_for_concept(args.concept_name, features_number)
            modifier.set_influence_factor_for_concept(args.concept_name, influence_factor)

            # Reset timestep counter for new generation
            modifier.reset_timestep()

            print(
                f"Unlearning: {args.concept_name} (features={features_number}, influence={influence_factor}, per_timestep={args.per_timestep})"
            )

            # Generate with unique seed for each image
            generator = torch.Generator(device).manual_seed(args.seed)
            inference_start = time.time()

            # Capture representations and generate image
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
                f"Image generated and {num_repr} representations captured in {inference_time:.2f} seconds"
            )

            # --------------------------------------------------------------------------
            # SAVE OUTPUTS FOR THIS PARAMETER PAIR
            # --------------------------------------------------------------------------
            # Determine output directory
            if args.output_dir:
                output_dir = Path(args.output_dir)
            else:
                output_dir = results_base_path / "images"

            output_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with parameter info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_prompt = "".join(
                c for c in args.prompt[:30] if c.isalnum() or c in (" ", "_")
            ).strip()
            save_prompt = save_prompt.replace(" ", "_")

            # Format: concept_name_fn{features}_if{influence}
            concept_name_clean = args.concept_name.replace(" ", "_")
            concept_suffix = (
                f"_unlearn_{concept_name_clean}_fn{features_number}_if{influence_factor}"
            )

            image_filename = f"task_{task_id}_{timestamp}_{save_prompt}{concept_suffix}.png"
            image_path = output_dir / image_filename

            # Save image
            image.save(image_path)
            print(f"✓ Image saved to: {image_path}")

            # Save representations
            repr_dir = output_dir / "representations"
            repr_dir.mkdir(parents=True, exist_ok=True)

            repr_base_filename = f"task_{task_id}_{timestamp}_{save_prompt}{concept_suffix}"
            saved_repr_files = []

            print(f"Saving {len(representations)} layer representations...")
            for i, (layer_path, repr_tensor) in enumerate(
                zip(layers_to_capture, representations, strict=True)
            ):
                layer_name = layer_path.name
                repr_filename = f"{repr_base_filename}_{layer_name}.pt"
                repr_file_path = repr_dir / repr_filename

                torch.save(repr_tensor, repr_file_path)
                saved_repr_files.append(repr_filename)

            print(f"✓ Representations saved to: {repr_dir}")

            # Save metadata
            metadata_file = output_dir / f"task_{task_id}_{timestamp}{concept_suffix}_metadata.txt"
            with open(metadata_file, "w") as f:
                f.write(f"Task {task_id}\n")
                f.write(f"Job: {job_id}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Prompt: {args.prompt}\n")
                f.write(f"Concept: {args.concept_name}\n")
                f.write(f"Features Number: {features_number}\n")
                f.write(f"Influence Factor: {influence_factor}\n")
                f.write(f"Per Timestep: {args.per_timestep}\n")
                f.write(f"Device: {device}\n")
                f.write(f"Guidance Scale: {args.guidance_scale}\n")
                f.write(f"Steps: {args.steps}\n")
                f.write(f"Seed: {args.seed + pair_idx}\n")
                f.write(f"Model Load Time: {model_load_time:.2f}s\n")
                f.write(f"Inference Time: {inference_time:.2f}s\n")
                f.write(f"Image: {image_filename}\n")
                f.write(f"\nCaptured Representations ({len(representations)} layers):\n")
                for i, (layer_path, repr_tensor) in enumerate(
                    zip(layers_to_capture, representations, strict=True)
                ):
                    f.write(f"  {i + 1}. {layer_path.name}: {tuple(repr_tensor.shape)}\n")
                    f.write(f"     File: {saved_repr_files[i]}\n")

            print(f"✓ Metadata saved: {metadata_file}")

        # Cleanup: detach modifier
        modifier.detach()

        # Final summary
        print("\n" + "=" * 50)
        print(f"Grid generation complete: {len(args.parameter_pairs)} images generated")
        print(f"Model load time: {model_load_time:.2f}s")
        print(f"Output directory: {output_dir}")
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
