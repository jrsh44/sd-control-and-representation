#!/usr/bin/env python3
"""
Generate unlearned images using SAE-based concept modification.

Usage:
  uv run scripts/sd_v1_5/generate_unlearned_cache_from_file.py \
    --dataset-name "test" --prompts-file /mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation/data/nudity/prompts_test.txt \
    --concept "exposed anus" \
    --results-dir /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/images \
    --model-name "SD_V1_5" --device "cpu" \
    --guidance-scale 7.5 --steps 50 --seed 42 \
    --sae-dir-path /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096 \
    --concept-sums-path /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/feature_merged/merged_feature_sums.pt \
    --epsilon 1e-8 \
    --influence-factors 1.0 2.0 3.0 \
    --feature-numbers 10 20 30

Note: Uses UNET_UP_1_ATT_1 layer for modifier attachment (hardcoded)

Output: {results_dir}/{concept}/fn{feature_number:02d}_if{influence_factor:.1f}/prompt_{id:04d}.png
"""

import argparse
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Dict

import torch
from dotenv import load_dotenv
from overcomplete.sae import TopKSAE

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from src.data.prompts import load_base_prompts  # noqa: E402
from src.models.config import ModelRegistry  # noqa: E402
from src.models.sd_v1_5.hooks import capture_layer_representations_with_unlearning  # noqa: E402
from src.models.sd_v1_5.layers import LayerPath  # noqa: E402
from src.utils.model_loader import ModelLoader  # noqa: E402
from src.utils.RepresentationModifier import RepresentationModifier  # noqa: E402
from src.utils.wandb import get_system_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Generate class-based or generic representations")

    # Required Config
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset (creates subfolder in cached_representations/)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="Path to prompts file (format: 'id;prompt' or just 'prompt')",
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        help="A concept to be unlearned",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: env RESULTS_DIR)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="SD_V1_5",
        choices=[model.name for model in ModelRegistry],
        help=(
            "Model to use (default: SD_V1_5). Options: "
            + ", ".join([model.name for model in ModelRegistry])
        ),
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument(
        "--log-images-every",
        type=int,
        default=None,
        help="Log generated images to WandB every N prompts (e.g., --log-images-every 10)",
    )
    parser.add_argument(
        "--sae-dir-path",
        type=str,
        required=True,
        help="Path to SAE weights directory",
    )
    parser.add_argument(
        "--concept-sums-path",
        type=str,
        required=True,
        help="Path to concept sums file",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Small value to avoid division by zero (default: 1e-8)",
    )
    parser.add_argument(
        "--generate_without_unlearning",
        action="store_true",
        help="If set, generates images without applying unlearning (for comparison)",
    )
    parser.add_argument(
        "--influence-factors",
        type=float,
        nargs="+",
        required=True,
        help="List of influence factors for unlearning (e.g., --influence-factors 1.0 2.0 3.0)",
    )
    parser.add_argument(
        "--feature-numbers",
        type=int,
        nargs="+",
        required=True,
        help="List of feature numbers to use for unlearning (e.g., --feature-numbers 10 20 30)",
    )
    parser.add_argument(
        "--per_timestep",
        action="store_true",
        help="If set, applies unlearning per timestep (default: per_timestep=False)",
    )
    return parser.parse_args()


def main():
    # --------------------------------------------------------------------------
    # 1. Parse Arguments and Setup
    # --------------------------------------------------------------------------
    args = parse_args()

    # Setup paths
    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        print(f"ERROR: Prompts file does not exist: {prompts_path}")
        return 1

    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif os.environ.get("RESULTS_DIR"):
        results_dir = Path(os.environ["RESULTS_DIR"])
    else:
        print(
            "ERROR: Results directory not specified. Use --results-dir or set RESULTS_DIR in .env"
        )
        return 1

    # Setup device
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # 2. Load Model
    # --------------------------------------------------------------------------
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    model_load_start = time.time()
    model_registry = ModelRegistry[args.model_name]

    loader = ModelLoader(model_enum=model_registry)
    pipe = loader.load_model(device=device)
    model_load_time = time.time() - model_load_start
    model_name = model_registry.config_name
    print(f"Model loaded: {model_name} in {model_load_time:.2f}s")

    # --------------------------------------------------------------------------
    # 3. Load SAE and Setup RepresentationModifier
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LOADING SAE AND SETTING UP UNLEARNING")
    print("=" * 80)
    sae_load_start = time.time()

    sae_dir_path = Path(args.sae_dir_path)
    if not sae_dir_path.exists():
        raise FileNotFoundError(f"SAE directory not found: {sae_dir_path}")

    # Find the first model.pt or checkpoint.pt file
    pt_files = list(sae_dir_path.glob("model.pt")) + list(sae_dir_path.glob("checkpoint.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No model.pt or checkpoint.pt file found in {sae_dir_path}")

    sae_path = pt_files[0]
    print(f"Using SAE weights: {sae_path}")

    # Extract topk from path
    match = re.search(r"topk(\d+)_.*_bs(\d+)", str(sae_path))
    if match:
        extracted_topk = int(match.group(1))
        print(f"Extracted top_k={extracted_topk} from SAE path")
    else:
        print("Warning: Could not extract top_k from SAE path")
        extracted_topk = 32
        print(f"Using default top_k={extracted_topk}")

    # Load SAE state dict
    sae_dict = torch.load(sae_path, map_location="cpu")
    state_dict = sae_dict.get("model_state_dict", sae_dict)

    # Remove _orig_mod prefix if present
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("Detected torch.compile() prefix → removing '_orig_mod.'")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Infer dimensions
    enc_weight = state_dict["encoder.final_block.0.weight"]
    input_shape = enc_weight.shape[1]
    nb_concepts = enc_weight.shape[0]
    print(f"Inferred input_dim={input_shape}, nb_concepts={nb_concepts}")

    # Create and load SAE
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

    # Load concept scores
    concept_scores_path = Path(args.concept_sums_path)
    if not concept_scores_path.exists():
        raise FileNotFoundError(f"Concept sums not found: {concept_scores_path}")
    concept_sums = torch.load(concept_scores_path, map_location=device)
    print(f"✓ Concept sums loaded from {concept_scores_path}")

    # Create RepresentationModifier
    modifier = RepresentationModifier(
        sae=sae,
        stats_dict=concept_sums,
        epsilon=args.epsilon,
        ignore_modification="true",
        device=device,
    )

    # Use hardcoded layer for modifier attachment
    sae_layer_path = LayerPath.UNET_UP_1_ATT_1
    modifier.attach_to(pipe, sae_layer_path)
    print(f"✓ Modifier attached to {sae_layer_path.name}")

    # Add concept to unlearn (initial parameters will be updated in generation loop)
    initial_influence = args.influence_factors[0]
    initial_features = args.feature_numbers[0]
    modifier.add_concept_to_unlearn(
        concept_name=args.concept,
        influence_factor=initial_influence,
        features_number=initial_features,
        per_timestep=args.per_timestep,
    )
    print(
        f"✓ Concept '{args.concept}' added for unlearning "
        f"(initial: influence={initial_influence}, features={initial_features})"
    )
    print(
        f"   Will test {len(args.influence_factors)} influence factors × "
        f"{len(args.feature_numbers)} feature numbers = "
        f"{len(args.influence_factors) * len(args.feature_numbers)} combinations"
    )

    sae_load_time = time.time() - sae_load_start
    print(f"SAE and modifier setup completed in {sae_load_time:.2f}s")

    # --------------------------------------------------------------------------
    # 4. Load Prompts
    # --------------------------------------------------------------------------
    print("\nLoading prompts...")
    raw_prompts = load_base_prompts(prompts_path)
    if not raw_prompts:
        print("ERROR: No prompts loaded from file")
        return 1

    # Replace {} with concept name in each prompt
    prompts: Dict[int, str] = {}
    for prompt_id, prompt_template in raw_prompts:
        if "{}" in prompt_template:
            prompts[prompt_id] = prompt_template.replace("{}", args.concept)
        else:
            prompts[prompt_id] = prompt_template

    total_prompts = len(prompts)
    print(f"Loaded {total_prompts} prompts for concept '{args.concept}'")
    print(f"Example prompt: {list(prompts.values())[0]}")

    # --------------------------------------------------------------------------
    # 5. Initialize WandB (Optional)
    # --------------------------------------------------------------------------
    if not args.skip_wandb:
        import wandb

        wandb.login()
        run_name = f"Unlearning_{args.concept.replace(' ', '_')}_{args.dataset_name}"

        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "Unknown"

        config = {
            "dataset": {
                "name": args.dataset_name,
                "prompts_file": str(prompts_path),
                "concept": args.concept,
                "total_prompts": total_prompts,
            },
            "model": {
                "name": model_name,
                "load_time": model_load_time,
            },
            "generation": {
                "guidance_scale": args.guidance_scale,
                "steps": args.steps,
                "base_seed": args.seed,
            },
            "unlearning": {
                "influence_factors": args.influence_factors,
                "feature_numbers": args.feature_numbers,
                "total_combinations": len(args.influence_factors) * len(args.feature_numbers),
            },
            "hardware": {
                "device": device,
                "gpu_name": gpu_name,
                "platform": platform.platform(),
            },
        }

        wandb.init(
            project="sd-control-representation",
            entity="bartoszjezierski28-warsaw-university-of-technology",
            name=run_name,
            config=config,
            group=f"unlearning_{args.concept.replace(' ', '_')}",
            job_type="generate_unlearned_images",
            tags=["unlearning", "generation", args.dataset_name, args.concept],
            notes=f"Image generation with unlearning for concept: {args.concept}",
        )
        print(f"🚀 WandB Initialized: {run_name}")

    # --------------------------------------------------------------------------
    # 6. Generation
    # --------------------------------------------------------------------------
    # Generation statistics
    total_generations = 0
    failed_generations = 0
    total_inference_time = 0.0
    total_combinations = len(args.influence_factors) * len(args.feature_numbers)

    # --------------------------------------------------------------------------
    # 6a. Generate baseline images without unlearning (if requested)
    # --------------------------------------------------------------------------
    if args.generate_without_unlearning:
        print("\n" + "=" * 80)
        print("GENERATING BASELINE IMAGES (NO INTERVENTION)")
        print("=" * 80)

        concept_dir = results_dir / args.concept.replace(" ", "_")
        baseline_dir = concept_dir / "no_intervention"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Baseline directory: {baseline_dir}")

        # Set modifier to ignore modifications for baseline generation
        modifier.set_ignore_modification()
        print("✓ Modifier set to ignore modifications (generating without unlearning)")

        for prompt_nr, prompt_text in prompts.items():
            # Check if image already exists
            image_filename = f"prompt_{prompt_nr:04d}.png"
            image_path = baseline_dir / image_filename

            if image_path.exists():
                print(f"  [{prompt_nr}] ⏭️  Skipping (already exists): {image_filename}")
                continue

            print(f"  [{prompt_nr}] 🎨 Generating baseline: {prompt_text[:60]}...")

            try:
                generator = torch.Generator(device).manual_seed(args.seed)
                system_metrics_start = get_system_metrics(device) if not args.skip_wandb else {}

                # Generate image without unlearning (modifier ignores modifications)
                inference_start = time.time()
                representations, image = capture_layer_representations_with_unlearning(
                    pipe=pipe,
                    prompt=prompt_text,
                    layer_paths=[],  # Empty list - no representations cached
                    modifier=modifier,  # Modifier with ignore_modification=true
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                )
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                # Save image to disk
                image.save(image_path)

                total_generations += 1
                print(f"      ✅ Generated in {inference_time:.2f}s, saved to {image_filename}")

                # Log to WandB
                if not args.skip_wandb:
                    system_metrics_end = get_system_metrics(device)
                    log_data = {
                        "concept": args.concept,
                        "baseline": True,
                        "prompt_nr": prompt_nr,
                        "inference_time": inference_time,
                        "total_generations": total_generations,
                        **system_metrics_end,
                        "gpu_memory_delta_mb": system_metrics_end.get("gpu_memory_mb", 0)
                        - system_metrics_start.get("gpu_memory_mb", 0),
                    }

                    if args.log_images_every and (total_generations % args.log_images_every == 0):
                        log_data["generated_image"] = wandb.Image(image, caption=prompt_text)

                    wandb.log(log_data)

                # Free GPU memory
                if image is not None:
                    del image
                torch.cuda.empty_cache()

            except Exception as e:
                if not args.skip_wandb:
                    wandb.log(
                        {
                            "error": str(e),
                            "concept": args.concept,
                            "baseline": True,
                            "prompt_nr": prompt_nr,
                            "prompt_text": prompt_text,
                        }
                    )
                failed_generations += 1
                print(f"      ❌ ERROR: {e}")
                import traceback

                traceback.print_exc()

        # Re-enable modifications for parameter combinations
        modifier.unset_ignore_modification()
        print("\n✓ Modifier re-enabled for parameter combinations")
        print("=" * 80)

    # --------------------------------------------------------------------------
    # 6b. Generate images with unlearning (parameter combinations)
    # --------------------------------------------------------------------------
    print(
        f"\nStarting generation for {total_prompts} prompts × "
        f"{total_combinations} parameter combinations..."
    )
    print("=" * 80)

    # Outer loop: parameter combinations
    for influence_factor in args.influence_factors:
        for feature_number in args.feature_numbers:
            print(f"\n{'=' * 80}")
            print(f"PARAMETER COMBINATION: influence={influence_factor}, features={feature_number}")
            print(f"{'=' * 80}")

            # Create output directory for this parameter combination
            concept_dir = results_dir / args.concept.replace(" ", "_")
            param_dir = concept_dir / f"fn{feature_number:02d}_if{influence_factor:.1f}"
            param_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Output directory: {param_dir}")

            # Update modifier parameters (cheap operation)
            modifier.set_influence_factor_for_concept(args.concept, influence_factor)
            modifier.set_number_of_features_for_concept(args.concept, feature_number)
            print("✓ Updated modifier parameters")

            # Inner loop: prompts
            for prompt_nr, prompt_text in prompts.items():
                # Check if image already exists
                image_filename = f"prompt_{prompt_nr:04d}.png"
                image_path = param_dir / image_filename

                if image_path.exists():
                    print(f"  [{prompt_nr}] ⏭️  Skipping (already exists): {image_filename}")
                    total_generations += 1
                    continue

                print(f"  [{prompt_nr}] 🎨 Generating: {prompt_text[:60]}...")

                try:
                    generator = torch.Generator(device).manual_seed(args.seed)
                    system_metrics_start = get_system_metrics(device) if not args.skip_wandb else {}

                    # Generate unlearned image without caching representations
                    inference_start = time.time()
                    representations, image = capture_layer_representations_with_unlearning(
                        pipe=pipe,
                        prompt=prompt_text,
                        layer_paths=[],  # Empty list - no representations cached
                        modifier=modifier,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                    )
                    inference_time = time.time() - inference_start
                    total_inference_time += inference_time

                    # Save image to disk (image_path already defined above)
                    image.save(image_path)

                    total_generations += 1
                    print(f"      ✅ Generated in {inference_time:.2f}s, saved to {image_filename}")

                    # Log to WandB
                    if not args.skip_wandb:
                        system_metrics_end = get_system_metrics(device)
                        log_data = {
                            "concept": args.concept,
                            "influence_factor": influence_factor,
                            "feature_number": feature_number,
                            "prompt_nr": prompt_nr,
                            "inference_time": inference_time,
                            "total_generations": total_generations,
                            **system_metrics_end,
                            "gpu_memory_delta_mb": system_metrics_end.get("gpu_memory_mb", 0)
                            - system_metrics_start.get("gpu_memory_mb", 0),
                        }

                        # Log image every N prompts if requested
                        if args.log_images_every and (
                            total_generations % args.log_images_every == 0
                        ):
                            log_data["generated_image"] = wandb.Image(image, caption=prompt_text)

                        wandb.log(log_data)

                    # Free GPU memory
                    if image is not None:
                        del image
                    torch.cuda.empty_cache()

                except Exception as e:
                    if not args.skip_wandb:
                        wandb.log(
                            {
                                "error": str(e),
                                "concept": args.concept,
                                "influence_factor": influence_factor,
                                "feature_number": feature_number,
                                "prompt_nr": prompt_nr,
                                "prompt_text": prompt_text,
                            }
                        )
                    failed_generations += 1
                    print(f"      ❌ ERROR: {e}")
                    import traceback

                    traceback.print_exc()
    # 7. Summary
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Concept: {args.concept}")
    print(f"Device: {device}")
    print(f"Model Load Time: {model_load_time:.2f}s")
    print(f"SAE Load Time: {sae_load_time:.2f}s")
    print(f"Total Inference Time: {total_inference_time:.2f}s")
    if total_generations > 0:
        print(f"Avg Inference Time: {total_inference_time / total_generations:.2f}s per generation")
    print("\nParameter Combinations:")
    print(f"  Influence Factors: {args.influence_factors}")
    print(f"  Feature Numbers: {args.feature_numbers}")
    print(f"  Total Combinations: {total_combinations}")
    print("\nResults:")
    print(f"  ✅ Generated: {total_generations}")
    print(f"  ❌ Failed: {failed_generations}")
    print("=" * 80)

    if not args.skip_wandb:
        wandb.log(
            {
                "final/total_generations": total_generations,
                "final/failed_generations": failed_generations,
                "final/model_load_time": model_load_time,
                "final/sae_load_time": sae_load_time,
                "final/total_inference_time": total_inference_time,
                "final/avg_inference_time": total_inference_time / max(total_generations, 1),
            }
        )

        wandb.finish()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
