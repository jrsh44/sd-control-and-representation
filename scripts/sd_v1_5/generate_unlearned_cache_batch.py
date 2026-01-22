#!/usr/bin/env python3
"""
uv run scripts/sd_v1_5/generate_unlearned_cache_batch.py \
    --results_dir /mnt/evafs/groups/mi2lab/jcwalina/results/test \
    --prompts_csv /mnt/evafs/groups/mi2lab/jcwalina/results/test/prompts/prompts.csv \
    --sae_dir_path /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp16_topk32_lr5em5_ep2_bs4096 \
    --concept_sums_path /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp16_topk32_lr5em5_ep2_bs4096/feature_merged/merged_feature_sums.pt \
    --epsilon 1e-8 \
    --ignore_modification false \
    --layers UNET_UP_1_ATT_1 UNET_DOWN_2_RES_0 \
    --device cpu \
    --guidance_scale 7.5 \
    --steps 50 \
    --seed 42 \
    --skip_wandb \
    --unlearn_concept "exposed anus" 140 25 \
    --unlearn_concept "buttocks" 100 20 \
    --skip_existence_check
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
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

from src.data.cache import RepresentationCache  # noqa: E402
from src.models.config import ModelRegistry  # noqa: E402
from src.models.sd_v1_5.hooks import capture_layer_representations_with_unlearning  # noqa: E402
from src.models.sd_v1_5.layers import (  # noqa: E402
    LayerPath,  # noqa: E402
)
from src.models.sd_v1_5.hooks import capture_layer_representations_with_unlearning  # noqa: E402

# from src.utils.model_loader import ModelLoader  # noqa: E402
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
        "--prompts_csv",
        type=str,
        required=True,
        help="Path to CSV file with prompts (columns: id, prompt)",
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
        help="Path to concept sums file",
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
        "--unlearn_concept",
        action="append",
        nargs=4,
        metavar=("CONCEPT_NAME", "INFLUENCE_FACTOR", "FEATURES_NUMBER", "PER_TIMESTEP"),
        help="Concept to unlearn with parameters. Can be specified multiple times. "
        "Example: --unlearn_concept 'exposed anus' 140 25 true",
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
    # --------------------------------------------------------------------------
    # 0. Parse arguments
    # --------------------------------------------------------------------------
    args = parse_args()

    # Setup paths
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    prompts_csv = Path(args.prompts_csv)
    if not prompts_csv.exists():
        print(f"ERROR: Prompts CSV does not exist: {prompts_csv}")
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

    # --------------------------------------------------------------------------
    # 1. MODEL
    # --------------------------------------------------------------------------
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

    # Load state dict
    sae_dict = torch.load(sae_path, map_location=device)
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
    sae_layer_path = LayerPath.UNET_UP_1_ATT_1  # hardcoded for test

    print("✓ SAE loaded and moved to device")

    # --------------------------------------------------------------------------
    # 3. Prepare modifier
    # --------------------------------------------------------------------------
    concept_scores_path = Path(args.concept_sums_path)
    if not concept_scores_path.exists():
        raise FileNotFoundError(f"Concept sums not found: {concept_scores_path}")

    concept_sums = torch.load(concept_scores_path, map_location=device)

    modifier = RepresentationModifier(
        sae=sae,
        stats_dict=concept_sums,
        epsilon=args.epsilon,
        ignore_modification=args.ignore_modification,
        device=device,
    )
    modifier.attach_to(pipe, sae_layer_path)

    # Add concepts to unlearn based on user arguments
    if args.unlearn_concept:
        print(f"\nAdding {len(args.unlearn_concept)} concept(s) to unlearn:")
        for concept_params in args.unlearn_concept:
            concept_name = concept_params[0]
            influence_factor = float(concept_params[1])
            features_number = int(concept_params[2])
            per_timestep = concept_params[3].lower() == "true"
            # Calculate scores (this also serves as a validation step)

            print(
                f"  - {concept_name}: influence={influence_factor}, features={features_number}, per_timestep={per_timestep}"  # noqa: E501
            )  # noqa: E501
            modifier.add_concept_to_unlearn(
                concept_name=concept_name,
                influence_factor=influence_factor,
                features_number=features_number,
                per_timestep=per_timestep,
            )
    else:
        print(
            "\nWarning: No concepts specified for unlearning. Images will be generated without modification."
        )

    # Build cache path
    model_name = model_registry.name
    cache_dir = results_dir / model_name / "unlearned_representations"

    # Print configuration
    print("=" * 70)
    print("Generate Unlearned Representations Cache (SD 1.5)")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Cache Dir: {cache_dir}")
    print(f"Prompts CSV: {prompts_csv}")
    print(f"Layers: {', '.join(layer.name for layer in layers_to_capture)}")
    print(f"Device: {device}")
    print("=" * 70)

    # Initialize memmap cache
    cache = RepresentationCache(cache_dir, use_fp16=True)

    # Load prompts from CSV
    print("\nLoading prompts from CSV...")
    try:
        prompts_df = pd.read_csv(prompts_csv)
        if "id" not in prompts_df.columns or "prompt" not in prompts_df.columns:
            print("ERROR: CSV must have 'id' and 'prompt' columns")
            return 1
        print(f"Loaded {len(prompts_df)} prompts from CSV")
    except Exception as e:
        print(f"ERROR: Failed to load CSV: {e}")
        return 1

    # Calculate dataset size for memmap cache
    print("\n📊 Calculating dataset size...")
    total_prompts = len(prompts_df)

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
                "guidance_scale": args.guidance_scale,
                "steps": args.steps,
                "base_seed": args.seed,
                "storage_format": "npy_memmap",
                "layers": [layer.name for layer in layers_to_capture],
                "unlearn_concepts": args.unlearn_concept if args.unlearn_concept else [],
                "sae_dir": str(args.sae_dir_path),
            },
            tags=[
                "memmap",
                "cache_generation",
                "unlearned",
                "sae",
            ],
        )

    # Generation statistics
    total_generations = 0
    skipped_generations = 0
    failed_generations = 0
    total_inference_time = 0.0
    total_save_time = 0.0
    generated = 0

    print("\nStarting unlearned generation...")
    print("=" * 70)

    for _, row in prompts_df.iterrows():
        prompt_id = row["id"]
        prompt_text = row["prompt"]

        # Use prompt_id as both object_name and prompt_nr for cache
        object_name = f"prompt_{prompt_id}"
        prompt_nr = 1  # Single prompt per ID

        # Check if all layers already have this representation
        if not args.skip_existence_check:
            all_exist = True
            for layer in layers_to_capture:
                existing_entries = cache.get_existing_entries(layer.name)
                if (prompt_nr, object_name, "") not in existing_entries:
                    all_exist = False
                    break

            if all_exist:
                skipped_generations += 1
                print(f"  [ID {prompt_id}] ⏭️  Skipping (already cached)")
                continue

        print(f"  [ID {prompt_id}] 🎨 Generating: {prompt_text[:60]}...")

        try:
            generator = torch.Generator(device).manual_seed(args.seed + prompt_id)
            system_metrics_start = get_system_metrics(device) if not args.skip_wandb else {}

            # Capture representations
            inference_start = time.time()
            representations, image = capture_layer_representations_with_unlearning(
                pipe=pipe,
                prompt=prompt_text,
                layer_paths=layers_to_capture,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                modifier=modifier,
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
                        print(f"     Shape per prompt: [{n_timesteps}, {n_spatial}, {n_features}]")
                        print(f"     Samples per prompt: {samples_per_prompt}")
                        print(f"     Estimated total: {total_samples_estimate:,}")

                        cache.initialize_layer(
                            layer_name=layer.name,
                            total_samples=total_samples_estimate,
                            feature_dim=n_features,
                        )

                    cache.save_representation(
                        layer_name=layer.name,
                        prompt_nr=prompt_nr,
                        prompt_text=prompt_text,
                        representation=tensor,
                        num_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        object_name=object_name,
                        style="",
                    )

            # Optional: save image
            img_dir = cache_dir / "images"
            img_dir.mkdir(exist_ok=True)
            img_path = img_dir / f"prompt_{prompt_id}.png"
            image.save(img_path)

            print(f"      Done in {inference_time:.2f}s → {img_path.name}")

            if not args.skip_wandb:
                wandb.log(
                    {
                        "prompt_id": prompt_id,
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
                        "prompt_id": prompt_id,
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
                        "prompt_id": prompt_id,
                        "prompt_text": prompt_text,
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
    print(f"Prompts CSV: {prompts_csv}")
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
