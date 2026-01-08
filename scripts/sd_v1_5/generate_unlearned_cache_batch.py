#!/usr/bin/env python3
"""
Generate cached representations for Stable Diffusion v1.5 using memmap format.

Structure: {results_dir}/{model_name}/cached_representations/{layer_name}/
Each dataset contains: object, style, prompt_nr, prompt_text, representation

EXAMPLE:
uv run scripts/sd_v1_5/generate_unlearned_cache_batch.py \
    --results_dir ./results \
    --prompts_dir ./data/prompts/validation \
    --sae_path ./models/sae/sae_weights.pt \
    --concept_means_path ./data/concept_means/concept_means.pt \
    --influence_factor 50.0 \
    --features_number 2 \
    --epsilon 1e-8 \
    --ignore_modification true \
    --layers UNET_UP_1_ATT_1 UNET_DOWN_2_RES_0 \
    --device cuda \
    --guidance_scale 7.5 \
    --steps 50 \
    --seed 42 \
    --skip_wandb \
    --skip_existence_check
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
import wandb
from diffusers import StableDiffusionPipeline  # noqa: E402
from dotenv import load_dotenv
from overcomplete.sae import TopKSAE

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from src.data import load_prompts_from_directory  # noqa: E402
from src.data.cache import RepresentationCache  # noqa: E402
from src.models.config import ModelRegistry  # noqa: E402
from src.models.sd_v1_5 import (  # noqa: E402
    LayerPath,  # noqa: E402
)
from src.models.sd_v1_5.hooks import capture_layer_representations_with_unlearning  # noqa: E402
from src.utils.RepresentationModifier import RepresentationModifier  # noqa: E402
from src.utils.wandb import get_system_metrics  # noqa: E402


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
        description="Generate cached representations using memmap format"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Results directory (default: $RESULTS_DIR from .env, required if not set)",
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        required=True,
        help="Path to directory containing .txt prompt files",
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
        default=5,
        help="Number of top features to unlearn",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon value for numerical stability",
    )
    parser.add_argument(
        "--ignore_modification",
        type=str,
        default="false",
        choices=["false", "no_sae", "raw_sae"],
        help="Type of generation modification",
    )
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        required=True,
        help="List of layer names to capture",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu, default: cuda)",
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_wandb", action="store_true")
    parser.add_argument(
        "--skip_existence_check",
        action="store_true",
        help="Skip checking if representations already exist (regenerate all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup paths
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    prompts_dir = Path(args.prompts_dir)
    if not prompts_dir.exists():
        print(f"ERROR: Prompts directory does not exist: {prompts_dir}")
        return 1

    layers_to_capture = parse_layer_names(args.layers)
    if not layers_to_capture:
        print("ERROR: No valid layers specified")
        return 1

    # Setup results directory with model_name/cached_representations subdirectory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif os.environ.get("RESULTS_DIR"):
        results_dir = Path(os.environ["RESULTS_DIR"])
    else:
        print("ERROR: Results directory not specified.")
        print("Please either:")
        print("  1. Use --results-dir argument, or")
        print("  2. Set RESULTS_DIR in .env file")
        return 1

    #############################################
    # MODEL
    #############################################
    print("\nLoading model...")
    model_load_start = time.time()
    model_registry = ModelRegistry.FINETUNED_SAEURON
    # loader = ModelLoader(model_enum=model_registry)
    # device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    # pipe = loader.load_model(device=device)
    model_id = "sd-legacy/stable-diffusion-v1-5"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f}s")

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
    sae = sae.to(device).to(dtype=torch.float32)
    sae.eval()
    sae_layer_path = LayerPath.UNET_UP_1_ATT_1
    print("SAE loaded and moved to device")

    #############################################
    # Prepare modifier
    #############################################
    concept_scores_path = Path(args.concept_means_path)
    if not concept_scores_path.exists():
        raise FileNotFoundError(f"Concept means not found: {concept_scores_path}")

    concept_means = torch.load(concept_scores_path, map_location="cpu")
    if (
        "sums_true_per_timestep" not in concept_means
        or "counts_true_per_timestep" not in concept_means
        or "sums_false_per_timestep" not in concept_means
        or "counts_false_per_timestep" not in concept_means
        or "timesteps" not in concept_means
    ):
        raise ValueError(
            "Concept means file must contain 'sums_true_per_timestep', 'counts_true_per_timestep', 'sums_false_per_timestep', 'counts_false_per_timestep', and 'timesteps' keys."  # noqa: E501
        )

    modifier = RepresentationModifier(
        sae=sae,
        stats_dict=concept_means,
        influence_factor=args.influence_factor,
        features_number=args.features_number,
        epsilon=args.epsilon,
        ignore_modification=args.ignore_modification,
    )
    modifier.attach_to(pipe, sae_layer_path)

    # Build cache path:
    # - With style: {results_dir}/{model_name}/cached_representations/
    # - No style: {results_dir}/{model_name}/validation/
    model_name = model_registry.name
    if args.style:
        cache_dir = results_dir / model_name / "cached_representations"
    else:
        cache_dir = results_dir / model_name / "validation"

    # Setup device
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    # Print configuration
    print("=" * 70)
    print("Generate Representations Cache (SD 1.5)")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Cache Dir: {cache_dir}")
    print(f"Cache Type: {'Validation (no style)' if not args.style else 'Training (with style)'}")
    print(f"Prompts: {prompts_dir}")
    print(f"Style: {args.style if args.style else 'None (validation)'}")
    print(f"Layers: {', '.join(layer.name for layer in layers_to_capture)}")
    print(f"Device: {device}")
    print("=" * 70)

    # Initialize memmap cache
    cache = RepresentationCache(cache_dir, use_fp16=True)

    # Load prompts
    print("\nLoading prompts...")
    prompts_by_object = load_prompts_from_directory(prompts_dir)
    if not prompts_by_object:
        print("ERROR: No prompts loaded")
        return 1

    # Calculate dataset size for memmap cache
    print("\n📊 Calculating dataset size...")
    # Count total prompts
    total_prompts = sum(len(prompts) for prompts in prompts_by_object.values())

    # Each prompt generates representations with shape [timesteps, 1, spatial, features]
    # We initialize layers dynamically after first generation to get actual dimensions
    print(f"  Total prompts: {total_prompts}")
    print(f"  Layers: {len(layers_to_capture)}")
    print("  NOTE: Will initialize layers dynamically after first generation")

    # Initialize wandb
    if not args.skip_wandb:
        wandb.login()
        wandb.init(
            project="sd-control-representation",
            entity="bartoszjezierski28-warsaw-university-of-technology",
            config={
                "model": model_name,
                "device": device,
                "style": args.style or "no_style",
                "guidance_scale": args.guidance_scale,
                "steps": args.steps,
                "base_seed": args.seed,
                "storage_format": "npy_memmap",
                "layers": [layer.name for layer in layers_to_capture],
            },
            tags=[
                "memmap",
                "cache_generation",
                args.style or "no_style",
                "sae",
                "feature_selection",
            ],
        )

    # Generation statistics
    total_generations = 0
    skipped_generations = 0
    failed_generations = 0
    total_inference_time = 0.0
    total_save_time = 0.0
    generated = 0

    style = args.style
    if style:
        print(f"\nStarting generation for style: {style}")
    else:
        print("\nStarting generation (no style applied)")
    print("=" * 70)

    for object_name, prompts in prompts_by_object.items():
        print(f"\n[Object: {object_name}] Processing {len(prompts)} prompts...")

        for prompt_nr, base_prompt in prompts.items():
            # Check if all layers already have this representation
            if not args.skip_existence_check:
                all_exist = all(
                    cache.check_exists(layer.name, object_name, style or "", prompt_nr)
                    for layer in layers_to_capture
                )

                if all_exist:
                    skipped_generations += 1
                    print(f"  [{prompt_nr}] ⏭️  Skipping (already cached)")
                    continue

            # Generate with optional style
            if style:
                styled_prompt = f"{base_prompt} in {style} style"
            else:
                styled_prompt = base_prompt
            print(f"  [{prompt_nr}] 🎨 Generating: {styled_prompt[:60]}...")

            try:
                generator = torch.Generator(device).manual_seed(args.seed + prompt_nr)
                system_metrics_start = get_system_metrics(device) if not args.skip_wandb else {}

                # Capture representations
                inference_start = time.time()
                representations, image = capture_layer_representations_with_unlearning(
                    pipe=pipe,
                    prompt=styled_prompt,
                    layers_to_capture=layers_to_capture,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    modifier=modifier,
                    device=device,
                )
                inference_time = time.time() - inference_start
                total_inference_time += inference_time
                generated += 1

                # Save each representation immediately
                save_start = time.time()
                for layer, tensor in zip(layers_to_capture, representations, strict=True):
                    if tensor is not None:
                        # Auto-initialize layer on first save
                        if layer.name not in cache._active_memmaps:
                            # Calculate total samples for this layer
                            # tensor shape: [timesteps, 1, spatial, features]
                            n_timesteps, _, n_spatial, n_features = tensor.shape
                            samples_per_prompt = n_timesteps * n_spatial
                            total_samples_estimate = total_prompts * samples_per_prompt

                            print(f"\n  📦 Initializing layer '{layer.name}'")
                            print(
                                f"     Shape per prompt: [{n_timesteps}, {n_spatial}, {n_features}]"
                            )
                            print(f"     Samples per prompt: {samples_per_prompt}")
                            print(f"     Estimated total: {total_samples_estimate:,}")

                            cache.initialize_layer(
                                layer_name=layer.name,
                                total_samples=total_samples_estimate,
                                feature_dim=n_features,
                            )

                        cache.save_representation(
                            layer_name=layer.name,
                            object_name=object_name,
                            style=style or "",
                            prompt_nr=prompt_nr,
                            prompt_text=styled_prompt,
                            representation=tensor,
                            num_steps=args.steps,
                            guidance_scale=args.guidance_scale,
                        )

                # Optional: save image
                img_dir = cache_dir / "images"
                img_dir.mkdir(exist_ok=True)
                img_path = img_dir / f"{object_name}_prompt{prompt_nr:04d}.png"
                image.save(img_path)

                print(f"      Done in {inference_time:.2f}s → {img_path.name}")

                if not args.skip_wandb:
                    wandb.log(
                        {
                            "object": object_name,
                            "prompt_nr": prompt_nr,
                            "inference_time": inference_time,
                            "generated": generated,
                        }
                    )

                save_time = time.time() - save_start
                total_save_time += save_time

                # Free GPU memory
                del representations
                if image is not None:
                    del image
                torch.cuda.empty_cache()

                total_generations += 1
                print(f"      ✅ Generated in {inference_time:.2f}s, saved in {save_time:.2f}s")

                if not args.skip_wandb:
                    system_metrics_end = get_system_metrics(device)
                    wandb.log(
                        {
                            "object": object_name,
                            "prompt_nr": prompt_nr,
                            "inference_time": inference_time,
                            "save_time": save_time,
                            "total_generations": total_generations,
                            "skipped_generations": skipped_generations,
                            **system_metrics_end,
                            "gpu_memory_delta_mb": system_metrics_end.get("gpu_memory_mb", 0)
                            - system_metrics_start.get("gpu_memory_mb", 0),
                        }
                    )

            except Exception as e:
                if not args.skip_wandb:
                    wandb.log(
                        {
                            "error": str(e),
                            "object": object_name,
                            "prompt_nr": prompt_nr,
                            "styled_prompt": styled_prompt,
                        }
                    )
                failed_generations += 1
                print(f"      ❌ ERROR: {e}")
                import traceback

                traceback.print_exc()

    # Save all accumulated metadata
    print("\n💾 Finalizing cache...")
    metadata_save_start = time.time()
    # Finalize all layers (trim memmaps, save index files)
    print("  Finalizing cache layers...")
    for layer in layers_to_capture:
        if layer.name in cache._active_memmaps:
            cache.finalize_layer(layer.name)
    metadata_save_time = time.time() - metadata_save_start
    print(f"✅ Cache finalized in {metadata_save_time:.2f}s")

    # Print summary
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    print(f"Style: {style if style else 'None'}")
    print(f"Device: {device}")
    print(f"Model Load Time: {model_load_time:.2f}s")
    print(f"Total Inference Time: {total_inference_time:.2f}s")
    print(f"Total Save Time: {total_save_time:.2f}s")
    print(f"Metadata Save Time: {metadata_save_time:.2f}s")
    if total_generations > 0:
        print(f"Avg Inference Time: {total_inference_time / total_generations:.2f}s per generation")
        print(f"Avg Save Time: {total_save_time / total_generations:.2f}s per generation")
    print("\nResults:")
    print(f"  ✅ Generated: {total_generations}")
    print(f"  ⏭️ Skipped: {skipped_generations}")
    print(f"  ❌ Failed: {failed_generations}")
    print(f"\nCache Location: {cache_dir}/")
    print("=" * 70)

    # Log final summary
    if not args.skip_wandb:
        wandb.log(
            {
                "final/total_generations": total_generations,
                "final/skipped_generations": skipped_generations,
                "final/failed_generations": failed_generations,
                "final/model_load_time": model_load_time,
                "final/total_inference_time": total_inference_time,
                "final/total_save_time": total_save_time,
                "final/metadata_save_time": metadata_save_time,
                "final/avg_inference_time": total_inference_time / max(total_generations, 1),
                "final/avg_save_time": total_save_time / max(total_generations, 1),
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
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
