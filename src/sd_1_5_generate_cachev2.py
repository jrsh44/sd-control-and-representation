#!/usr/bin/env python3
"""
Generate cached representations for Stable Diffusion 1_5 with multiple objects and styles.
Saves each representation separately in: results/sd_1_5/{layer_name}/{object}/{style}/{prompt_nr}.pt
"""

"""
EXAMPLE USAGE:
uv run src/sd_1_5_generate_cachev2.py \
    --prompts-dir data/unlearn_canvas/prompts/test \
    --style Impressionism \
    --layers TEXT_EMBEDDING_FINAL UNET_MID_ATT UNET_DOWN_1_ATT_1 UNET_DOWN_2_ATT_1 UNET_UP_2_ATT_2 UNET_UP_1_ATT_2
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

# Add project root to path to allow imports to work from any location
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env from project root
load_dotenv(dotenv_path=project_root / ".env")

from diffusers import StableDiffusionPipeline
from src.utils.reprezentation import LayerPath, capture_layer_representations


def get_representation_path(
    base_dir: Path, layer_name: str, object_name: str, style: str, prompt_nr: int
) -> Path:
    """
    Get the path for a specific representation file.

    Args:
        base_dir: Base directory for all results
        layer_name: Name of the layer
        object_name: Name of the object
        style: Style name
        prompt_nr: Prompt number

    Returns:
        Path object for the representation file
    """
    return base_dir / layer_name / object_name / style / f"{prompt_nr}.pt"


def is_representation_cached(rep_path: Path) -> bool:
    """
    Check if a representation exists at the given path.

    Args:
        rep_path: Path to the representation file

    Returns:
        True if file exists, False otherwise
    """
    return rep_path.exists()


def save_representation(rep_path: Path, tensor: torch.Tensor):
    """
    Save a representation tensor to a file.

    Args:
        rep_path: Path where to save the tensor
        tensor: Representation tensor to save
    """
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, rep_path)


def load_representation(rep_path: Path) -> torch.Tensor:
    """
    Load a representation tensor from a file.

    Args:
        rep_path: Path to the representation file

    Returns:
        Loaded tensor or None if file doesn't exist
    """
    if rep_path.exists():
        try:
            return torch.load(rep_path, map_location="cpu")
        except Exception as e:
            print(f"Warning: Could not load representation from {rep_path}: {e}")
    return None


def load_prompts_from_directory(prompts_dir: Path) -> dict:
    """
    Load prompts from .txt files in the specified directory.
    Each file represents one object, and each line has format: ID; prompt
    Skips entries with empty prompts.

    Args:
        prompts_dir: Path to directory containing .txt files

    Returns:
        Dictionary mapping object names to dict of {prompt_id: prompt_text}
    """
    prompts_by_object = {}

    txt_files = sorted(prompts_dir.glob("*.txt"))

    for txt_file in txt_files:
        object_name = txt_file.stem.replace("sd_prompt_", "")

        prompts = {}
        skipped = 0
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if ";" in line:
                    parts = line.split(";", 1)
                    prompt_id = int(parts[0].strip())
                    prompt_text = parts[1].strip() if len(parts) > 1 else ""

                    if not prompt_text:
                        skipped += 1
                        continue

                    prompts[prompt_id] = prompt_text

        if prompts:
            prompts_by_object[object_name] = prompts
            status = f"  Loaded {len(prompts)} prompts for object: {object_name}"
            if skipped > 0:
                status += f" (skipped {skipped} empty)"
            print(status)

    return prompts_by_object


def parse_layer_names(layer_names: list) -> list:
    """
    Convert string layer names to LayerPath enum values.
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
    # ... [Same argument parsing as original] ...
    parser = argparse.ArgumentParser(
        description="Generate cached representations for multiple objects and styles"
    )
    parser.add_argument(
        "--prompts-dir",
        type=str,
        required=True,
        help="Path to directory containing .txt prompt files (one file per object)",
    )
    parser.add_argument(
        "--style",
        type=str,
        required=True,
        help="Style name from styles.txt (e.g., 'Impressionism', 'Van_Gogh')",
    )
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        required=True,
        help="List of layer names to capture (e.g., TEXT_EMBEDDING_FINAL UNET_MID_ATT)",
    )
    parser.add_argument(
        "--preferred-device",
        type=str,
        default="cuda",
        help="Preferred device (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup paths and device
    prompts_dir = Path(args.prompts_dir)
    if not prompts_dir.exists():
        print(f"ERROR: Prompts directory does not exist: {prompts_dir}")
        return 1

    layers_to_capture = parse_layer_names(args.layers)
    if not layers_to_capture:
        print("ERROR: No valid layers specified")
        return 1

    # Get results directory
    results_base_dir = os.environ.get("RESULTS_DIR")
    if results_base_dir:
        results_base_path = Path(results_base_dir)
    else:
        results_base_path = project_root / "results"

    base_output_dir = results_base_path / "sd_1_5"

    # Setup device
    device = "cuda" if args.preferred_device == "cuda" and torch.cuda.is_available() else "cpu"

    # Print configuration
    print("=" * 70)
    print("Cache Generation v2 - Stable Diffusion Representations")
    print("=" * 70)
    print(f"[Prompts Dir: {prompts_dir}]")
    print(f"[Style: {args.style}]")
    print(f"[Layers: {len(layers_to_capture)}]")
    print(f"[Device: {device}]")
    print(f"[Output Dir: {base_output_dir}]")
    print("=" * 70)

    # Load prompts
    print("\nLoading prompts...")
    prompts_by_object = load_prompts_from_directory(prompts_dir)
    if not prompts_by_object:
        print("ERROR: No prompts loaded")
        return 1

    # Load model
    print("\nLoading model...")
    model_id = "sd-legacy/stable-diffusion-v1-5"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_load_start = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    model_load_time = time.time() - model_load_start

    # Generation statistics
    total_generations = 0
    skipped_generations = 0
    failed_generations = 0

    style = args.style
    print(f"\nStarting generation for style: {style}")
    print("=" * 70)

    for object_name, prompts in prompts_by_object.items():
        print(f"\n[Object: {object_name}] Processing {len(prompts)} prompts...")

        for prompt_nr, base_prompt in prompts.items():
            # Check if all representations exist
            all_exist = all(
                is_representation_cached(
                    get_representation_path(
                        base_output_dir, layer.name, object_name, style, prompt_nr
                    )
                )
                for layer in layers_to_capture
            )

            if all_exist:
                skipped_generations += 1
                print(f"  [{prompt_nr}/{len(prompts)}] Skipping (already cached)")
                continue

            # Generate with style
            styled_prompt = f"{base_prompt} in {style} style"
            print(f"  [{prompt_nr}/{len(prompts)}] Generating: {styled_prompt[:60]}...")

            try:
                # Generate with fixed seed
                generator = torch.Generator(device).manual_seed(args.seed + prompt_nr)

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

                # Save each representation
                saved_count = 0
                for layer, tensor in zip(layers_to_capture, representations, strict=True):
                    if tensor is not None:
                        rep_path = get_representation_path(
                            base_output_dir, layer.name, object_name, style, prompt_nr
                        )
                        save_representation(rep_path, tensor)
                        saved_count += 1

                total_generations += 1
                print(
                    f"      ✓ Done in {inference_time:.2f}s ({saved_count} representations saved)"
                )

            except Exception as e:
                failed_generations += 1
                print(f"      ✗ ERROR: {e}")
                import traceback

                traceback.print_exc()

    # Print summary
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    print(f"Style: {style}")
    print(f"Device: {device}")
    print(f"Model Load Time: {model_load_time:.2f}s")
    print(f"\nResults:")
    print(f"  Generated: {total_generations}")
    print(f"  Skipped: {skipped_generations}")
    print(f"  Failed: {failed_generations}")
    print("\nOutput Structure:")
    print(f"  {base_output_dir}/")
    print(f"    └── <layer_name>/")
    print(f"        └── <object>/")
    print(f"            └── <style>/")
    print(f"                └── <prompt_nr>.pt")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        sys.exit(1)
