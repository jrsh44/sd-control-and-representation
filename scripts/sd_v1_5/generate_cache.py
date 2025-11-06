#!/usr/bin/env python3
"""
Generate cached representations for Stable Diffusion v1.5.

Structure: {results_dir}/cache/sd_v1_5/{layer_name}/
Each dataset contains: object, style, prompt_nr, prompt_text, representation

EXAMPLE:
uv run scripts/sd_v1_5/generate_cache.py \
    --prompts-dir data/unlearn_canvas/prompts/test \
    --style Impressionism \
    --layers TEXT_EMBEDDING_FINAL UNET_UP_3_ATT_2 UNET_UP_2_ATT_2 \
    --batch-size 10 \
    --skip-wandb
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
import wandb
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from diffusers import StableDiffusionPipeline  # noqa: E402

from src.data import RepresentationCache, load_prompts_from_directory  # noqa: E402
from src.models.sd_v1_5 import LayerPath, capture_layer_representations  # noqa: E402
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate cached representations using Arrow format"
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
        "--style",
        type=str,
        required=True,
        help="Style name (e.g., 'Impressionism', 'Van_Gogh')",
    )
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        required=True,
        help="List of layer names to capture",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of representations to batch before saving (default: 10)",
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

    args = parser.parse_args()

    # Setup paths
    prompts_dir = Path(args.prompts_dir)
    if not prompts_dir.exists():
        print(f"ERROR: Prompts directory does not exist: {prompts_dir}")
        return 1

    layers_to_capture = parse_layer_names(args.layers)
    if not layers_to_capture:
        print("ERROR: No valid layers specified")
        return 1

    # Setup results directory with cache/sd_1_5 subdirectory
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

    # Add cache/sd_1_5 subdirectory structure
    results_dir = results_dir / "cache" / "sd_1_5"

    # Setup device
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    # Print configuration
    print("=" * 70)
    print("Generate Representations Cache (SD 1.5)")
    print("=" * 70)
    print(f"Results Dir: {results_dir}")
    print(f"Prompts: {prompts_dir}")
    print(f"Style: {args.style}")
    print(f"Layers: {', '.join(layer.name for layer in layers_to_capture)}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {device}")
    print("=" * 70)

    # Initialize cache
    cache = RepresentationCache(results_dir)

    # Load prompts
    print("\nLoading prompts...")
    prompts_by_object = load_prompts_from_directory(prompts_dir)
    if not prompts_by_object:
        print("ERROR: No prompts loaded")
        return 1

    # Load model
    print("\nLoading model...")
    model_id = "sd-legacy/stable-diffusion-v1-5"
    variant = "fp16" if device == "cuda" else "fp32"

    model_load_start = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        variant=variant,
        safety_checker=None,
    ).to(device)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f}s")

    # Initialize wandb
    if not args.skip_wandb:
        wandb.login()
        wandb.init(
            project="sd-control-representation",
            entity="bartoszjezierski28-warsaw-university-of-technology",
            config={
                "model": model_id,
                "device": device,
                "style": args.style,
                "batch_size": args.batch_size,
                "guidance_scale": args.guidance_scale,
                "steps": args.steps,
                "base_seed": args.seed,
                "storage_format": "arrow",
                "layers": [layer.name for layer in layers_to_capture],
            },
            tags=["arrow", "cache_generation", args.style],
        )

    # Generation statistics
    total_generations = 0
    skipped_generations = 0
    failed_generations = 0
    total_inference_time = 0.0

    # No batch buffers - save immediately after each generation!

    style = args.style
    print(f"\nStarting generation for style: {style}")
    print("=" * 70)

    for object_name, prompts in prompts_by_object.items():
        print(f"\n[Object: {object_name}] Processing {len(prompts)} prompts...")

        for prompt_nr, base_prompt in prompts.items():
            # ============================================================
            # EXISTENCE CHECK - Comment out this block to disable
            # ============================================================
            # all_exist = all(
            #     cache.check_exists(layer.name, object_name, style, prompt_nr)
            #     for layer in layers_to_capture
            # )
            #
            # if all_exist:
            #     skipped_generations += 1
            #     print(f"  [{prompt_nr}] ⏭️  Skipping (already cached)")
            #     continue
            # ============================================================

            # Generate with style
            styled_prompt = f"{base_prompt} in {style} style"
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

                # Save immediately - one sample at a time, no batching
                save_start = time.time()
                for layer, tensor in zip(layers_to_capture, representations, strict=True):
                    if tensor is not None:
                        cache.save_batch(
                            layer.name,
                            [
                                {
                                    "object": object_name,
                                    "style": style,
                                    "prompt_nr": prompt_nr,
                                    "prompt_text": styled_prompt,
                                    "representation": tensor,
                                }
                            ],
                            verbose=False,  # Reduce log spam
                        )
                        # Free memory immediately after saving
                        del tensor

                # Free everything
                del representations
                if image is not None:
                    del image
                torch.cuda.empty_cache()  # Clear GPU cache

                save_time = time.time() - save_start

                total_generations += 1
                print(f"      ✅ Done in {inference_time:.2f}s | Saved in {save_time:.2f}s")

                # Log to wandb
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

    # Print summary
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    print(f"Style: {style}")
    print(f"Device: {device}")
    print(f"Model Load Time: {model_load_time:.2f}s")
    print(f"Total Inference Time: {total_inference_time:.2f}s")
    if total_generations > 0:
        print(f"Avg Inference Time: {total_inference_time / total_generations:.2f}s per generation")
    print("\nResults:")
    print(f"  ✅ Generated: {total_generations}")
    print(f"  ⏭️ Skipped: {skipped_generations}")
    print(f"  ❌ Failed: {failed_generations}")
    print(f"\nResults Location: {results_dir}/")
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
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
