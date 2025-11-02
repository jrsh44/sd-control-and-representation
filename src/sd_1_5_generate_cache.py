#!/usr/bin/env python3
"""
Generate cached representations for Stable Diffusion 1_5 with multiple objects and styles.
Saves each layer separately in: results/sd_1_5/{layer_name}/cached_rep.pt
Structure per layer: {object: {style: {prompt_nr: tensor[timesteps, ...]}}}
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

from diffusers import StableDiffusionPipeline  # noqa: E402

from src.utils.reprezentation import LayerPath, capture_layer_representations  # noqa: E402


def load_layer_cache(layer_path: Path) -> dict:
    """
    Load existing cache for a specific layer.
    
    Args:
        layer_path: Path to the layer's cached_rep.pt file
        
    Returns:
        Dictionary with structure {object: {style: {prompt_nr: tensor}}}
        Empty dict if file doesn't exist
    """
    if layer_path.exists():
        try:
            cache = torch.load(layer_path, map_location='cpu')
            return cache
        except Exception as e:
            print(f"Warning: Could not load existing cache from {layer_path}: {e}")
            return {}
    return {}


def save_layer_cache(layer_path: Path, cache: dict):
    """
    Save cache for a specific layer.
    
    Args:
        layer_path: Path to the layer's cached_rep.pt file
        cache: Dictionary with structure {object: {style: {prompt_nr: tensor}}}
    """
    layer_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, layer_path)


def is_combination_cached(cache: dict, object_name: str, style: str, prompt_nr: int) -> bool:
    """
    Check if a specific combination already exists in cache.
    
    Args:
        cache: Layer cache dictionary
        object_name: Object name
        style: Style name
        prompt_nr: Prompt number
        
    Returns:
        True if combination exists, False otherwise
    """
    return (
        object_name in cache
        and style in cache[object_name]
        and prompt_nr in cache[object_name][style]
    )


def add_to_cache(cache: dict, object_name: str, style: str, prompt_nr: int, tensor: torch.Tensor):
    """
    Add a new entry to cache.
    
    Args:
        cache: Layer cache dictionary
        object_name: Object name
        style: Style name
        prompt_nr: Prompt number
        tensor: Representation tensor
    """
    if object_name not in cache:
        cache[object_name] = {}
    if style not in cache[object_name]:
        cache[object_name][style] = {}
    cache[object_name][style][prompt_nr] = tensor


def clean_cache_empty_prompts(cache: dict, valid_prompts: dict, object_name: str, style: str) -> int:
    """
    Remove entries from cache where prompt_nr doesn't have a valid prompt.
    
    Args:
        cache: Layer cache dictionary
        valid_prompts: Dictionary of {prompt_nr: prompt_text} with only valid prompts
        object_name: Object name to clean
        style: Style name to clean
        
    Returns:
        Number of entries removed
    """
    removed = 0
    
    if object_name not in cache:
        return 0
    
    if style not in cache[object_name]:
        return 0
    
    # Get all prompt numbers currently in cache for this object+style
    cached_prompt_nrs = list(cache[object_name][style].keys())
    
    # Remove any that aren't in valid_prompts
    for prompt_nr in cached_prompt_nrs:
        if prompt_nr not in valid_prompts:
            del cache[object_name][style][prompt_nr]
            removed += 1
    
    # Clean up empty dicts
    if not cache[object_name][style]:
        del cache[object_name][style]
    if not cache[object_name]:
        del cache[object_name]
    
    return removed


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
        # Extract object name from filename (e.g., sd_prompt_Cats.txt -> Cats)
        object_name = txt_file.stem.replace("sd_prompt_", "")
        
        prompts = {}
        skipped = 0
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse format: ID; prompt
                if ';' in line:
                    parts = line.split(';', 1)
                    prompt_id = int(parts[0].strip())
                    prompt_text = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Skip empty prompts
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
    
    Args:
        layer_names: List of layer name strings (e.g., ["TEXT_EMBEDDING_FINAL", "UNET_MID_ATT"])
        
    Returns:
        List of LayerPath enum instances
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

    # Convert prompts directory to Path
    prompts_dir = Path(args.prompts_dir)
    if not prompts_dir.exists():
        print(f"ERROR: Prompts directory does not exist: {prompts_dir}")
        return 1

    # Parse layer names to LayerPath enums
    layers_to_capture = parse_layer_names(args.layers)
    if not layers_to_capture:
        print("ERROR: No valid layers specified")
        return 1

    print(f"\nLayers to capture: {[layer.name for layer in layers_to_capture]}")

    # Get SLURM environment variables
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job_id")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "no_slurm_task_id")

    # Get results directory from environment variable or use default
    results_base_dir = os.environ.get("RESULTS_DIR")
    if results_base_dir:
        results_base_path = Path(results_base_dir)
    else:
        # Default to relative path from project root
        results_base_path = project_root / "results"

    # Set device
    device = "cuda" if args.preferred_device == "cuda" and torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Cache Generation - Stable Diffusion Representations")
    print("=" * 70)
    print(f"[Job {job_id} | Task {task_id}]")
    print(f"[Prompts Dir: {prompts_dir}]")
    print(f"[Style: {args.style}]")
    print(f"[Layers: {len(layers_to_capture)}]")
    print(f"[Preferred Device: {args.preferred_device}]")
    print(f"[Actual Device: {device}]")
    print(f"[Guidance Scale: {args.guidance_scale}]")
    print(f"[Steps: {args.steps}]")
    print(f"[Seed: {args.seed}]")
    print(f"[Results Base Dir: {results_base_path}]")
    print("=" * 70)

    #############################################
    # LOAD PROMPTS
    #############################################
    print("\nLoading prompts from directory...")
    prompts_by_object = load_prompts_from_directory(prompts_dir)
    
    if not prompts_by_object:
        print("ERROR: No prompts loaded from directory")
        return 1
    
    total_prompts = sum(len(prompts) for prompts in prompts_by_object.values())
    print(f"\n✓ Loaded {len(prompts_by_object)} objects with {total_prompts} total prompts")

    #############################################
    # MODEL
    #############################################
    model_id = "sd-legacy/stable-diffusion-v1-5"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("\nLoading model...")
    model_load_start = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    model_load_time = time.time() - model_load_start
    print(f"✓ Model loaded in {model_load_time:.2f} seconds")

    #############################################
    # SETUP LAYER CACHES
    #############################################
    # Determine base output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = results_base_path / "sd_1_5"
    
    # Load existing caches for each layer
    layer_caches = {}
    layer_paths_dict = {}  # Store file paths for each layer
    
    print(f"\nBase output directory: {base_output_dir}")
    print("Loading existing caches...")
    
    total_cleaned = 0
    for layer in layers_to_capture:
        layer_name = layer.name
        layer_dir = base_output_dir / layer_name
        layer_file = layer_dir / "cached_rep.pt"
        
        layer_paths_dict[layer_name] = layer_file
        layer_caches[layer_name] = load_layer_cache(layer_file)
        
        cached_count = sum(
            len(prompts)
            for obj_cache in layer_caches[layer_name].values()
            for prompts in obj_cache.values()
        )
        print(f"  {layer_name}: {cached_count} existing entries")
    
    # Clean caches: remove entries with empty prompts
    print("\nCleaning caches (removing empty prompts)...")
    style = args.style
    
    for object_name, valid_prompts in prompts_by_object.items():
        for layer in layers_to_capture:
            layer_name = layer.name
            removed = clean_cache_empty_prompts(
                layer_caches[layer_name],
                valid_prompts,
                object_name,
                style
            )
            if removed > 0:
                total_cleaned += removed
                print(f"  {object_name} ({layer_name}): removed {removed} invalid entries")
    
    if total_cleaned > 0:
        print(f"✓ Cleaned {total_cleaned} total invalid entries")
    else:
        print("✓ No invalid entries found")
    
    #############################################
    # GENERATION & REPRESENTATION CAPTURING
    #############################################
    style = args.style
    total_generations = 0
    skipped_generations = 0
    failed_generations = 0
    
    print(f"\nStarting generation for style: {style}")
    print("=" * 70)
    
    for object_name, prompts in prompts_by_object.items():
        print(f"\n[Object: {object_name}] Processing {len(prompts)} prompts...")
        
        for prompt_nr, base_prompt in prompts.items():
            # Check if ALL layers already have this combination cached
            all_cached = all(
                is_combination_cached(layer_caches[layer.name], object_name, style, prompt_nr)
                for layer in layers_to_capture
            )
            
            if all_cached:
                skipped_generations += 1
                print(f"  [{prompt_nr}/{len(prompts)}] Skipping (already cached)")
                continue
            
            # Add style suffix to prompt
            styled_prompt = f"{base_prompt} in {style} style"
            
            print(f"  [{prompt_nr}/{len(prompts)}] Generating: {styled_prompt[:60]}...")
            
            try:
                # Set seed for this specific generation
                generator = torch.Generator(device).manual_seed(args.seed + prompt_nr)
                
                # Generate and capture representations
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
                
                # Store representations in respective layer caches
                layers_updated = 0
                for layer, repr_tensor in zip(layers_to_capture, representations, strict=True):
                    if repr_tensor is not None:
                        layer_name = layer.name
                        add_to_cache(
                            layer_caches[layer_name],
                            object_name,
                            style,
                            prompt_nr,
                            repr_tensor
                        )
                        layers_updated += 1
                
                total_generations += 1
                print(f"      ✓ Done in {inference_time:.2f}s ({layers_updated} layers updated)")
                
            except Exception as e:
                failed_generations += 1
                print(f"      ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "=" * 70)
    print(f"Generation complete:")
    print(f"  Generated: {total_generations}")
    print(f"  Skipped (cached): {skipped_generations}")
    print(f"  Failed: {failed_generations}")
    print("=" * 70)

    #############################################
    # SAVE CACHES
    #############################################
    print(f"\nSaving layer caches...")
    
    for layer in layers_to_capture:
        layer_name = layer.name
        layer_file = layer_paths_dict[layer_name]
        cache = layer_caches[layer_name]
        
        # Count entries in this layer
        total_entries = sum(
            len(prompts)
            for obj_cache in cache.values()
            for prompts in obj_cache.values()
        )
        
        print(f"  Saving {layer_name}: {total_entries} entries to {layer_file}")
        save_layer_cache(layer_file, cache)
    
    print(f"✓ All layer caches saved")
    
    #############################################
    # PRINT SUMMARY
    #############################################
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Job: {job_id} | Task: {task_id}")
    print(f"Style: {style}")
    print(f"Device: {device}")
    print(f"Guidance Scale: {args.guidance_scale}")
    print(f"Steps: {args.steps}")
    print(f"Base Seed: {args.seed}")
    print(f"Model Load Time: {model_load_time:.2f}s")
    
    print(f"\nGeneration Results:")
    print(f"  Generated: {total_generations}")
    print(f"  Skipped (cached): {skipped_generations}")
    print(f"  Failed: {failed_generations}")
    print(f"  Cleaned (empty prompts): {total_cleaned}")
    
    print(f"\nLayers Saved ({len(layers_to_capture)}):")
    for layer in layers_to_capture:
        layer_name = layer.name
        cache = layer_caches[layer_name]
        total_entries = sum(
            len(prompts)
            for obj_cache in cache.values()
            for prompts in obj_cache.values()
        )
        print(f"  {layer_name}: {total_entries} entries → {layer_paths_dict[layer_name]}")
    
    print(f"\nObjects Processed:")
    for object_name, prompts in prompts_by_object.items():
        print(f"  {object_name}: {len(prompts)} prompts")
    
    print("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        sys.exit(1)
