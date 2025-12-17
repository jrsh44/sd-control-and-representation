#!/usr/bin/env python3
"""
Generate cached representations from object prompts with optional artistic style.

Usage:
  uv run scripts/sd_v1_5/cache_rep_from_objects_dir_and_style_.py \
    --prompts-dir data/unlearn_canvas/prompts/test \
    --style Impressionism --layers UNET_UP_1_ATT_1 \
    --model-name sd_v1_5 --log-images-every 1

Output: {results_dir}/{model_name}/{dataset_name}/representations/{train|validation}/{layer_name}/
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import List

import torch
from dotenv import load_dotenv

import wandb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from src.data.cache import RepresentationCache  # noqa: E402
from src.data.prompts import load_prompts_from_directory  # noqa: E402
from src.models.config import ModelRegistry  # noqa: E402
from src.models.sd_v1_5.hooks import capture_layer_representations  # noqa: E402
from src.models.sd_v1_5.layers import LayerPath  # noqa: E402
from src.utils.model_loader import ModelLoader  # noqa: E402
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
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate cached representations using memmap format"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: $RESULTS_DIR from .env, required if not set)",
    )
    parser.add_argument(
        "--prompts-dir",
        type=str,
        required=True,
        help="Path to directory containing .txt prompt files",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="unlearn_canvas",
        help="Name of the dataset (used for organization and WandB logging)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        help="Style name (e.g., 'Impressionism', 'Van_Gogh'). "
        "If not provided, no style suffix is added.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        required=True,
        help="List of layer names to capture",
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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu, default: cuda)",
    )
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument(
        "--skip-existence-check",
        action="store_true",
        help="Skip checking if representations already exist (regenerate all)",
    )
    parser.add_argument(
        "--log-images-every",
        type=int,
        default=None,
        help="Log generated images to WandB every N prompts (e.g., --log-images-every 10)",
    )
    return parser.parse_args()


def main():
    # --------------------------------------------------------------------------
    # 1. Parse Arguments and Setup
    # --------------------------------------------------------------------------
    args = parse_args()

    # Setup paths
    prompts_dir = Path(args.prompts_dir)
    if not prompts_dir.exists():
        print(f"ERROR: Prompts directory does not exist: {prompts_dir}")
        return 1

    layers_to_capture = parse_layer_names(args.layers)
    if not layers_to_capture:
        print("ERROR: No valid layers specified")
        return 1

    # Setup results directory with model_name/dataset_name/representations subdirectory
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

    # Setup device
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # 2. Load Model
    # --------------------------------------------------------------------------
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    model_load_start = time.time()
    try:
        model_registry = ModelRegistry[args.model_name]
    except KeyError:
        print(f"ERROR: Invalid model name '{args.model_name}'")
        print(f"Available models: {', '.join([m.name for m in ModelRegistry])}")
        return 1
    loader = ModelLoader(model_enum=model_registry)
    pipe = loader.load_model(device=device)
    model_load_time = time.time() - model_load_start
    model_name = model_registry.config_name
    print(f"Model loaded: {model_name} in {model_load_time:.2f}s")

    # --------------------------------------------------------------------------
    # 3. Setup Cache Paths
    # --------------------------------------------------------------------------
    # Build cache path:
    # - With style: {results_dir}/{model_name}/{dataset_name}/representations/train/
    # - No style: {results_dir}/{model_name}/{dataset_name}/representations/validation/
    if args.style:
        cache_dir = results_dir / model_name / args.dataset_name / "representations" / "train"
    else:
        cache_dir = results_dir / model_name / args.dataset_name / "representations" / "validation"

    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Cache Dir: {cache_dir}")
    print(f"Cache Type: {'Validation (no style)' if not args.style else 'Training (with style)'}")
    print(f"Prompts: {prompts_dir}")
    print(f"Style: {args.style if args.style else 'None (validation)'}")
    print(f"Layers: {', '.join(layer.name for layer in layers_to_capture)}")
    print(f"Device: {device}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 4. Setup Cache and Load Prompts
    # --------------------------------------------------------------------------
    cache = RepresentationCache(cache_dir, use_fp16=True)

    print("\nLoading prompts...")
    prompts_by_object = load_prompts_from_directory(prompts_dir)
    if not prompts_by_object:
        print("ERROR: No prompts loaded")
        return 1

    total_prompts = sum(len(prompts) for prompts in prompts_by_object.values())

    print(f"  Total prompts: {total_prompts}")
    print(f"  Layers: {len(layers_to_capture)}")

    # --------------------------------------------------------------------------
    # 5. Initialize WandB
    # --------------------------------------------------------------------------
    if not args.skip_wandb:
        wandb.login()

        # Create a run name
        style_suffix = f"_{args.style}" if args.style else "_validation"
        run_name = (
            f"Cache_{args.dataset_name}_{model_name}_{style_suffix}_{len(layers_to_capture)}layers"
        )

        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "Unknown"

        # Structured Configuration
        config = {
            "dataset": {
                "name": args.dataset_name,
                "prompts_dir": str(prompts_dir),
                "style": args.style or "no_style",
                "cache_type": "validation" if not args.style else "training",
                "cache_dir": str(cache_dir),
                "total_prompts": total_prompts,
                "layers": [layer.name for layer in layers_to_capture],
            },
            "model": {
                "name": model_name,
                "registry": model_registry.config_name,
                "load_time": model_load_time,
            },
            "generation": {
                "guidance_scale": args.guidance_scale,
                "steps": args.steps,
                "base_seed": args.seed,
                "storage_format": "npy_memmap",
                "use_fp16": True,
            },
            "hardware": {
                "device": device,
                "gpu_name": gpu_name,
                "platform": platform.platform(),
                "python_version": sys.version.split()[0],
                "node": platform.node(),
            },
        }

        wandb.init(
            project="sd-control-representation",
            entity="bartoszjezierski28-warsaw-university-of-technology",
            name=run_name,
            config=config,
            group=f"cache_{args.dataset_name}_{model_name}",
            job_type="generate_cache",
            tags=["cache", "generation", "memmap", args.dataset_name, args.style or "validation"]
            + [layer.name.lower() for layer in layers_to_capture],
            notes="Generated cached representations for SD layers using memmap format.",
        )
        print(f"🚀 WandB Initialized: {run_name}")

    # --------------------------------------------------------------------------
    # 6. Generation
    # --------------------------------------------------------------------------
    # Load existing (prompt_nr, object_name) tuples ONCE at start for efficient batch filtering
    # We use get_existing_entries because prompt_nr alone is not unique across objects
    existing_entries: set = set()
    if not args.skip_existence_check:
        print("\n🔍 Checking for existing representations...")
        for layer in layers_to_capture:
            layer_existing = cache.get_existing_entries(layer.name)
            if not existing_entries:
                existing_entries = layer_existing
            else:
                # Only skip if ALL layers have it
                existing_entries = existing_entries.intersection(layer_existing)
        print(f"  Found {len(existing_entries)} already cached (prompt_nr, object) entries")

    # Generation statistics
    total_generations = 0
    skipped_generations = 0
    failed_generations = 0
    total_inference_time = 0.0
    total_save_time = 0.0

    style = args.style
    if style:
        print(f"\nStarting generation for style: {style}")
    else:
        print("\nStarting generation (no style applied)")
    print("=" * 80)

    for object_name, prompts in prompts_by_object.items():
        print(f"\n[Object: {object_name}] Processing {len(prompts)} prompts...")

        for prompt_nr, base_prompt in prompts.items():
            # Check existence using pre-loaded set with (prompt_nr, object_name) key
            entry_key = (prompt_nr, object_name)
            if entry_key in existing_entries:
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
                representations, image = capture_layer_representations(
                    pipe=pipe,
                    prompt=styled_prompt,
                    layer_paths=layers_to_capture,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                )
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                # Save each representation
                save_start = time.time()
                for layer, tensor in zip(layers_to_capture, representations, strict=True):
                    if tensor is not None:
                        # Auto-initialize layer on first save
                        if layer.name not in cache._active_memmaps:
                            # Calculate total samples for this layer
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
                            prompt_nr=prompt_nr,
                            prompt_text=styled_prompt,
                            representation=tensor,
                            num_steps=args.steps,
                            guidance_scale=args.guidance_scale,
                            object_name=object_name,
                            style=style or "",
                        )
                save_time = time.time() - save_start
                total_save_time += save_time

                total_generations += 1
                print(f"      ✅ Generated in {inference_time:.2f}s, saved in {save_time:.2f}s")

                # Log to WandB before freeing memory
                if not args.skip_wandb:
                    system_metrics_end = get_system_metrics(device)
                    log_data = {
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

                    # Log image every N prompts if requested
                    if args.log_images_every and (total_generations % args.log_images_every == 0):
                        log_data["generated_image"] = wandb.Image(image, caption=styled_prompt)

                    wandb.log(log_data)

                # Free GPU memory
                del representations
                if image is not None:
                    del image
                torch.cuda.empty_cache()

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

    # --------------------------------------------------------------------------
    # 7. Finalize and Save
    # --------------------------------------------------------------------------
    # Save all accumulated metadata
    print("\n💾 Finalizing cache...")
    metadata_save_start = time.time()
    print("  Finalizing cache layers...")
    for layer in layers_to_capture:
        if layer.name in cache._active_memmaps:
            cache.finalize_layer(layer.name)
    metadata_save_time = time.time() - metadata_save_start
    print(f"✅ Cache finalized in {metadata_save_time:.2f}s")

    print("\n" + "=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)
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
    print("=" * 80)

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
