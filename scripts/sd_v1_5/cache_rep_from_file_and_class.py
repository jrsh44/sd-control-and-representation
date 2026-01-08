#!/usr/bin/env python3
"""
Generate cached representations using class templates or direct prompts.

Usage:
  uv run scripts/sd_v1_5/cache_rep_from_file_and_class.py \
    --dataset-name "cars" --prompts-file data/cars/templates.txt \
    --classes "sedan" "suv" --layers UNET_UP_1_ATT_1

Output: {results_dir}/{model_name}/{dataset_name}/representations/{layer_name}/
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from src.data.cache import RepresentationCache  # noqa: E402
from src.data.prompts import (  # noqa: E402
    build_prompts_by_class,
    load_base_prompts,
    load_classes_file,
)
from src.models.config import ModelRegistry  # noqa: E402
from src.models.sd_v1_5.hooks import capture_layer_representations  # noqa: E402
from src.models.sd_v1_5.layers import LayerPath  # noqa: E402
from src.utils.model_loader import ModelLoader  # noqa: E402
from src.utils.wandb import get_system_metrics  # noqa: E402


def parse_layer_names(layer_names: List[str]) -> List[LayerPath]:
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
        "--layers",
        type=str,
        nargs="+",
        required=True,
        help="List of layer names to capture",
    )

    # Class Configuration (Optional)
    parser.add_argument(
        "--classes-file",
        type=str,
        default=None,
        help="Path to classes.txt (optional)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="List of class names to process (e.g., --classes 'dog' 'cat')",
    )

    # Generation Config
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
    parser.add_argument("--skip-existence-check", action="store_true")
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

    # Parse layers
    layers_to_capture = parse_layer_names(args.layers)
    if not layers_to_capture:
        print("ERROR: No valid layers specified")
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
    # 3. Setup Cache Paths
    # --------------------------------------------------------------------------
    # Build cache path: {results_dir}/{model_name}/{dataset_name}/representations
    cache_dir = results_dir / model_name / args.dataset_name / "representations"
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Cache Dir: {cache_dir}")
    print(f"Prompts File: {prompts_path}")
    print(f"Layers: {', '.join(layer.name for layer in layers_to_capture)}")
    print(f"Device: {device}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 4. Setup Cache and Load Prompts
    # --------------------------------------------------------------------------
    cache = RepresentationCache(cache_dir, use_fp16=True)

    print("\nLoading prompts...")
    # Helper to load raw prompts (handling id;text format)
    # Using load_base_prompts from src.data.prompts as it handles the ; parsing
    raw_prompts = load_base_prompts(prompts_path)
    if not raw_prompts:
        print("ERROR: No prompts loaded from file")
        return 1

    prompts_by_group: Dict[str, Dict[int, str]] = {}

    # Logic: Determine if we are doing Class-Filling or Direct Generation
    has_classes = args.classes is not None or args.classes_file is not None

    if has_classes:
        # --- Mode 1: Class-Based Generation ---
        all_classes = {}

        # Load from CLI
        if args.classes:
            print(f"Using {len(args.classes)} classes from command line")
            # Create a mock dict for consistency with load_classes_file output
            all_classes = dict(enumerate(args.classes))

        # Load from File (merging if both provided, though usually mutually exclusive usage)
        if args.classes_file:
            path = Path(args.classes_file)
            if not path.exists():
                print(f"ERROR: Classes file not found: {path}")
                return 1
            file_classes = load_classes_file(path)
            if file_classes:
                # If we already have CLI classes, append to them
                start_idx = len(all_classes)
                for i, cls in file_classes.items():
                    all_classes[start_idx + i] = cls

        if not all_classes:
            print("ERROR: Classes configuration provided but no classes found.")
            return 1

        # Use helper to explode templates X classes
        # This handles the "{}" replacement logic
        selected_ids = sorted(all_classes.keys())
        prompts_by_group = build_prompts_by_class(raw_prompts, all_classes, selected_ids)
        print(f"Mode: Class-Based ({len(raw_prompts)} templates × {len(all_classes)} classes)")

    else:
        # --- Mode 2: Direct Generation ---
        # Treat the dataset_name as the single "group"
        # Convert list of tuples [(id, text)] to dict {id: text}
        prompts_dict = dict(raw_prompts)
        prompts_by_group[args.dataset_name] = prompts_dict
        print(f"Mode: Direct Generation ({len(prompts_dict)} prompts)")

    total_prompts = sum(len(p) for p in prompts_by_group.values())
    print(f"  Total prompts: {total_prompts}")
    print(f"  Groups: {len(prompts_by_group)}")

    # --------------------------------------------------------------------------
    # 5. Initialize WandB
    # --------------------------------------------------------------------------
    if not args.skip_wandb:
        import wandb

        wandb.login()
        run_name = f"Cache_{args.dataset_name}_{model_name}_{len(layers_to_capture)}layers"

        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "Unknown"

        # Structured Configuration
        config = {
            "dataset": {
                "name": args.dataset_name,
                "prompts_file": str(prompts_path),
                "mode": "class_based" if has_classes else "direct",
                "num_groups": len(prompts_by_group),
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
            tags=["cache", "generation", "memmap", args.dataset_name]
            + [layer.name.lower() for layer in layers_to_capture],
            notes="Generated cached representations for SD layers from prompts file.",
        )
        print(f"🚀 WandB Initialized: {run_name}")

    # --------------------------------------------------------------------------
    # 6. Generation
    # --------------------------------------------------------------------------
    # Load existing (prompt_nr, object, style) tuples ONCE at start
    # We use get_existing_entries because prompt_nr alone is not unique across classes/styles
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
        print(f"  Found {len(existing_entries)} already cached (prompt_nr, class, style) entries")

    # Generation statistics
    total_generations = 0
    skipped_generations = 0
    failed_generations = 0
    total_inference_time = 0.0
    total_save_time = 0.0

    print(f"\nStarting generation for {total_prompts} prompts...")
    print("=" * 80)

    # Style is empty string by default (can be extended later)
    current_style = ""

    for group_label, prompts in prompts_by_group.items():
        # group_label is Class Name (e.g. "dog") or Dataset Name (if direct mode)
        print(f"\n[Group: {group_label}] Processing {len(prompts)} prompts...")

        for prompt_nr, prompt_text in prompts.items():
            # Check existence using (prompt_nr, object_name, style) key
            entry_key = (prompt_nr, group_label, current_style)
            if entry_key in existing_entries:
                skipped_generations += 1
                print(f"  [{prompt_nr}] ⏭️  Skipping (already cached)")
                continue

            print(f"  [{prompt_nr}] 🎨 Generating: {prompt_text[:60]}...")

            try:
                generator = torch.Generator(device).manual_seed(args.seed + prompt_nr)
                system_metrics_start = get_system_metrics(device) if not args.skip_wandb else {}

                # Capture representations
                inference_start = time.time()
                representations, image = capture_layer_representations(
                    pipe=pipe,
                    prompt=prompt_text,
                    layer_paths=layers_to_capture,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    skip_initial_timestep=True,
                )
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                # Save each representation
                save_start = time.time()
                for layer, tensor in zip(layers_to_capture, representations, strict=True):
                    if tensor is not None:
                        # Auto-initialize layer on first save
                        if layer.name not in cache._active_memmaps:
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
                            prompt_text=prompt_text,
                            representation=tensor,
                            num_steps=args.steps,
                            guidance_scale=args.guidance_scale,
                            object_name=group_label,
                        )
                save_time = time.time() - save_start
                total_save_time += save_time

                total_generations += 1
                print(f"      ✅ Generated in {inference_time:.2f}s, saved in {save_time:.2f}s")

                # Log to WandB before freeing memory
                if not args.skip_wandb:
                    system_metrics_end = get_system_metrics(device)
                    log_data = {
                        "group": group_label,
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
                        log_data["generated_image"] = wandb.Image(image, caption=prompt_text)

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
                            "group": group_label,
                            "prompt_nr": prompt_nr,
                            "prompt_text": prompt_text,
                        }
                    )
                failed_generations += 1
                print(f"      ❌ ERROR: {e}")
                import traceback

                traceback.print_exc()

    # --------------------------------------------------------------------------
    # 7. Finalize and Save
    # --------------------------------------------------------------------------
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
    print(f"Dataset: {args.dataset_name}")
    print(f"Mode: {'Class-Based' if has_classes else 'Direct'}")
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
