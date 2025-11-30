#!/usr/bin/env python3
"""
Generate cached representations from a single prompt file with prompt_nr;prompt format.

EXAMPLE:
# Generate from 100 random prompts
uv run scripts/sd_v1_5/cache_rep_from_file.py \
    --prompts-file data/cc3m-wds/train.txt \
    --num-prompts 100 \
    --layers TEXT_EMBEDDING_FINAL UNET_UP_1_ATT_1 \
    --skip-wandb

# SLURM array job: split into 10 jobs
uv run scripts/sd_v1_5/cache_rep_from_file.py \
    --prompts-file data/cc3m-wds/train.txt \
    --num-prompts 1000 \
    --array-id 2 \
    --array-total 10 \
    --layers UNET_UP_1_ATT_1 \
    --log-images-every 1


# Process specific prompt range
uv run scripts/sd_v1_5/cache_rep_from_file.py \
    --prompts-file data/cc3m-wds/train.txt \
    --prompt-range 1 100 \
    --layers UNET_UP_1_ATT_1 \
    --log-images-every 1
"""

import argparse
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from dotenv import load_dotenv

import wandb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from src.data.cache import RepresentationCache  # noqa: E402
from src.data.prompts import load_base_prompts  # noqa: E402
from src.models.config import ModelRegistry  # noqa: E402
from src.models.sd_v1_5.hooks import capture_layer_representations  # noqa: E402
from src.models.sd_v1_5.layers import LayerPath  # noqa: E402
from src.utils.model_loader import ModelLoader  # noqa: E402
from src.utils.wandb import get_system_metrics  # noqa: E402


def select_prompts_for_job(
    all_prompts: Dict[int, str],
    num_prompts: int = None,
    prompt_range: tuple = None,
    array_id: int = None,
    array_total: int = None,
    seed: int = 42,
) -> Dict[int, str]:
    """
    Select prompts based on various filters.

    Priority:
    1. If prompt_range is specified, use that range
    2. If array_id/array_total are specified, split into chunks first
    3. If num_prompts is specified, sample randomly from the chunk
    4. Otherwise, use all prompts (or all from chunk)

    Args:
        all_prompts: All available prompts
        num_prompts: Number of random prompts to select from the chunk
        prompt_range: Tuple of (start, end) prompt numbers (inclusive)
        array_id: SLURM array task ID (0-indexed)
        array_total: Total number of array jobs
        seed: Random seed for sampling

    Returns:
        Selected prompts dictionary
    """
    # Priority 1: Specific range
    if prompt_range is not None:
        start, end = prompt_range
        selected = {nr: prompt for nr, prompt in all_prompts.items() if start <= nr <= end}
        print(f"Selected prompts in range [{start}, {end}]: {len(selected)} prompts")
        return selected

    # Priority 2: Array job splitting (split first, then sample)
    working_set = all_prompts
    if array_id is not None and array_total is not None:
        # Sort prompt numbers for consistent splitting
        sorted_nrs = sorted(all_prompts.keys())

        # Calculate chunk size
        total_count = len(sorted_nrs)
        chunk_size = total_count // array_total
        remainder = total_count % array_total

        # Calculate start and end indices for this job
        # Distribute remainder evenly among first jobs
        if array_id < remainder:
            start_idx = array_id * (chunk_size + 1)
            end_idx = start_idx + chunk_size + 1
        else:
            start_idx = array_id * chunk_size + remainder
            end_idx = start_idx + chunk_size

        # Select prompts for this chunk
        chunk_nrs = sorted_nrs[start_idx:end_idx]
        working_set = {nr: all_prompts[nr] for nr in chunk_nrs}

        print(f"Array job {array_id + 1}/{array_total}:")
        print(f"  Chunk size: {len(working_set)} prompts (indices {start_idx}-{end_idx - 1})")
        if chunk_nrs:
            print(f"  Prompt range: {min(chunk_nrs)} - {max(chunk_nrs)}")

    # Priority 3: Random sampling from working set
    if num_prompts is not None:
        if num_prompts >= len(working_set):
            print(
                f"Requested {num_prompts} prompts, "
                f"but only {len(working_set)} available in chunk. Using all."
            )
            return working_set
        else:
            random.seed(seed)
            selected_nrs = random.sample(list(working_set.keys()), num_prompts)
            selected = {nr: working_set[nr] for nr in selected_nrs}
            print(f"  Randomly sampled {len(selected)} prompts from chunk (seed={seed})")
            return selected

    # Priority 4: Use all from working set
    if array_id is not None:
        print(f"  Using all {len(working_set)} prompts from chunk")
    else:
        print(f"Using all {len(working_set)} prompts")
    return working_set


def parse_layer_names(layer_names: List[str]) -> List[LayerPath]:
    """Convert string layer names to LayerPath enum values."""
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
    parser = argparse.ArgumentParser(description="Generate cached representations from prompt file")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: $RESULTS_DIR from .env)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="Path to prompts file with format: prompt_nr;prompt",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="cc3m-wds",
        help="Dataset name for cache path organization (default: cc3m-wds)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        required=True,
        help="List of layer names to capture",
    )

    # Prompt selection options (mutually exclusive groups)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--num-prompts",
        type=int,
        help="Number of random prompts to select",
    )
    selection_group.add_argument(
        "--prompt-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Process prompts in range [START, END] (inclusive)",
    )

    # Array job support
    parser.add_argument(
        "--array-id",
        type=int,
        help="SLURM array task ID (0-indexed). Use with --array-total",
    )
    parser.add_argument(
        "--array-total",
        type=int,
        help="Total number of array jobs. Use with --array-id",
    )

    # Generation options
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

    # Validate array job arguments
    if (args.array_id is not None) != (args.array_total is not None):
        print("ERROR: --array-id and --array-total must be used together")
        return 1

    if args.array_id is not None and args.array_id >= args.array_total:
        print(f"ERROR: --array-id ({args.array_id}) must be < --array-total ({args.array_total})")
        return 1

    # Setup paths
    prompts_file = Path(args.prompts_file)
    if not prompts_file.exists():
        print(f"ERROR: Prompts file does not exist: {prompts_file}")
        return 1

    layers_to_capture = parse_layer_names(args.layers)
    if not layers_to_capture:
        print("ERROR: No valid layers specified")
        return 1

    # Setup results directory
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
    model_registry = ModelRegistry.FINETUNED_SAEURON
    loader = ModelLoader(model_enum=model_registry)
    pipe = loader.load_model(device=device)
    model_load_time = time.time() - model_load_start
    model_name = model_registry.name
    print(f"Model loaded: {model_name} in {model_load_time:.2f}s")

    # --------------------------------------------------------------------------
    # 3. Setup Cache Paths
    # --------------------------------------------------------------------------
    cache_dir = results_dir / model_name / args.dataset_name / "representations" / "train"

    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Cache Dir: {cache_dir}")
    print(f"Prompts File: {prompts_file}")
    print(f"Layers: {', '.join(layer.name for layer in layers_to_capture)}")
    print(f"Device: {device}")
    if args.array_id is not None:
        print(f"Array Job: {args.array_id + 1}/{args.array_total}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 4. Setup Cache and Load Prompts
    # --------------------------------------------------------------------------
    cache = RepresentationCache(cache_dir, use_fp16=True)

    print("\nLoading prompts...")
    raw_prompts = load_base_prompts(prompts_file)
    all_prompts = dict(raw_prompts)
    print(f"Loaded {len(all_prompts)} prompts from {prompts_file}")

    selected_prompts = select_prompts_for_job(
        all_prompts=all_prompts,
        num_prompts=args.num_prompts,
        prompt_range=tuple(args.prompt_range) if args.prompt_range else None,
        array_id=args.array_id,
        array_total=args.array_total,
        seed=args.seed,
    )

    if not selected_prompts:
        print("ERROR: No prompts selected")
        return 1

    total_prompts = len(selected_prompts)

    print(f"  Selected prompts: {total_prompts}")
    print(f"  Layers: {len(layers_to_capture)}")

    # --------------------------------------------------------------------------
    # 5. Initialize WandB
    # --------------------------------------------------------------------------
    if not args.skip_wandb:
        wandb.login()

        # Create a run name
        run_name = f"Cache_{args.dataset_name}_{model_name}_{len(layers_to_capture)}layers"

        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "Unknown"

        # Structured Configuration
        config = {
            "dataset": {
                "name": args.dataset_name,
                "prompts_file": str(prompts_file),
                "object_name": "",
                "style": "no_style",
                "cache_dir": str(cache_dir),
                "total_prompts": len(all_prompts),
                "selected_prompts": total_prompts,
                "layers": [layer.name for layer in layers_to_capture],
            },
            "model": {
                "name": model_name,
                "registry": model_registry.name,
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

        if args.array_id is not None:
            config["dataset"]["array_id"] = args.array_id
            config["dataset"]["array_total"] = args.array_total

        wandb.init(
            project="sd-control-representation",
            entity="bartoszjezierski28-warsaw-university-of-technology",
            name=run_name,
            config=config,
            group=f"cache_{args.dataset_name}_{model_name}",
            job_type="generate_cache",
            tags=[
                "cache",
                "generation",
                args.dataset_name,
            ]
            + [layer.name.lower() for layer in layers_to_capture],
            notes="Generated cached representations for SD layers from prompt file.",
        )
        print(f"🚀 WandB Initialized: {run_name}")

    # --------------------------------------------------------------------------
    # 6. Generation
    # --------------------------------------------------------------------------
    # Generation statistics
    total_generations = 0
    skipped_generations = 0
    failed_generations = 0
    total_inference_time = 0.0
    total_save_time = 0.0

    print(f"\nStarting generation for {total_prompts} prompts...")
    print("=" * 80)

    for prompt_nr, prompt in sorted(selected_prompts.items()):
        # Check if all layers already have this representation
        if not args.skip_existence_check:
            all_exist = all(
                cache.check_exists(layer.name, "", "", prompt_nr) for layer in layers_to_capture
            )

            if all_exist:
                skipped_generations += 1
                print(f"  [{prompt_nr}] ⏭️  Skipping (already cached)")
                continue

        # Generate with optional style
        print(f"  [{prompt_nr}] 🎨 Generating: {prompt[:60]}...")

        try:
            generator = torch.Generator(device).manual_seed(args.seed + prompt_nr)
            system_metrics_start = get_system_metrics(device) if not args.skip_wandb else {}

            # Capture representations
            inference_start = time.time()
            representations, image = capture_layer_representations(
                pipe=pipe,
                prompt=prompt,
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
                        object_name="",
                        style="",
                        prompt_nr=prompt_nr,
                        prompt_text=prompt,
                        representation=tensor,
                        num_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                    )
            save_time = time.time() - save_start
            total_save_time += save_time

            total_generations += 1
            print(f"      ✅ Generated in {inference_time:.2f}s, saved in {save_time:.2f}s")

            if not args.skip_wandb:
                system_metrics_end = get_system_metrics(device)
                log_data = {
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
                    log_data["generated_image"] = wandb.Image(image, caption=prompt)

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
                        "prompt_nr": prompt_nr,
                        "styled_prompt": prompt,
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
    print(f"Device: {device}")
    if args.array_id is not None:
        print(f"Array Job: {args.array_id + 1}/{args.array_total}")
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
