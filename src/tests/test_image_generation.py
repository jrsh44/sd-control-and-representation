#!/usr/bin/env python3
"""
Test script for image generation with Stable Diffusion on cluster.
Generates images from text prompts and saves them to the results directory.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv

# Add project root to path to allow imports to work from any location
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env from project root
load_dotenv(dotenv_path=project_root / ".env")

from src.utils.reprezentation import LayerPath, capture_layer_representations  # noqa: E402


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion on cluster")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument(
        "--preferred-device",
        type=str,
        default="cuda",
        help="Preferred device (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (optional, overrides default)",
    )

    args = parser.parse_args()

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
    print(f"Model loaded in {model_load_time:.2f} seconds")

    #############################################
    # GENERATION & REPRESENTATION CAPTURING
    #############################################
    layers_to_capture = [
        # Text conditioning
        LayerPath.TEXT_EMBEDDING_FINAL,
        # Critical attention layers
        LayerPath.UNET_MID_ATT,
        LayerPath.UNET_DOWN_2_ATT_0,
        LayerPath.UNET_UP_1_ATT_2,
        # ResNet features for comparison
        LayerPath.UNET_DOWN_1_RES_0,
        LayerPath.UNET_MID_RES_1,
        LayerPath.UNET_UP_0_RES_2,
    ]

    generator = torch.Generator(device).manual_seed(args.seed)

    print("\nGenerating image and capturing representations...")
    inference_start = time.time()

    # Capture representations and generate image simultaneously
    representations, image = capture_layer_representations(
        pipe=pipe,
        prompt=args.prompt,
        layer_paths=layers_to_capture,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    inference_time = time.time() - inference_start
    num_repr = len(representations)
    print(
        f"Image generated and {num_repr} representations captured in {inference_time:.2f} seconds"
    )

    #############################################
    # SAVE OUTPUTS
    #############################################
    # Determine output directory
    if args.output_dir:
        # Use custom output directory if specified
        output_dir = Path(args.output_dir)
    else:
        # Use RESULTS_DIR/images or default results/images
        output_dir = results_base_path / "images"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {output_dir}")

    # Create filename with timestamp and task info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(c for c in args.prompt[:30] if c.isalnum() or c in (" ", "_")).strip()
    safe_prompt = safe_prompt.replace(" ", "_")

    image_filename = f"task_{task_id}_{timestamp}_{safe_prompt}.png"
    image_path = output_dir / image_filename

    # Save image
    image.save(image_path)
    print(f"\n✓ Image saved to: {image_path}")

    # Save representations
    repr_dir = output_dir / "representations"
    repr_dir.mkdir(parents=True, exist_ok=True)

    repr_base_filename = f"task_{task_id}_{timestamp}_{safe_prompt}"
    saved_repr_files = []

    print(f"\nSaving {len(representations)} layer representations...")
    for i, (layer_path, repr_tensor) in enumerate(
        zip(layers_to_capture, representations, strict=True)
    ):
        layer_name = layer_path.name
        repr_filename = f"{repr_base_filename}_{layer_name}.pt"
        repr_file_path = repr_dir / repr_filename

        # Save tensor
        torch.save(repr_tensor, repr_file_path)
        saved_repr_files.append(repr_filename)

        # Print shape info
        shape_str = f"{tuple(repr_tensor.shape)}"
        print(f"  [{i + 1}/{len(representations)}] {layer_name}: {shape_str} → {repr_filename}")

    print(f"\n✓ All representations saved to: {repr_dir}")

    # Save metadata
    metadata_file = output_dir / f"task_{task_id}_{timestamp}_metadata.txt"
    with open(metadata_file, "w") as f:
        f.write(f"Task {task_id}\n")
        f.write(f"Job: {job_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Guidance Scale: {args.guidance_scale}\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Model Load Time: {model_load_time:.2f}s\n")
        f.write(f"Inference Time: {inference_time:.2f}s\n")
        f.write(f"Total Time: {model_load_time + inference_time:.2f}s\n")
        f.write(f"Image: {image_filename}\n")
        f.write(f"\nCaptured Representations ({len(representations)} layers):\n")
        for i, (layer_path, repr_tensor) in enumerate(
            zip(layers_to_capture, representations, strict=True)
        ):
            f.write(f"  {i + 1}. {layer_path.name}: {tuple(repr_tensor.shape)}\n")
            f.write(f"     File: {saved_repr_files[i]}\n")

    print(f"✓ Metadata saved to: {metadata_file}")
    print("\nTiming Summary:")
    print(f"  Model Load: {model_load_time:.2f}s")
    print(f"  Inference: {inference_time:.2f}s")
    print(f"  Total: {model_load_time + inference_time:.2f}s")
    print("\nOutput Summary:")
    print(f"  Image: {image_path}")
    print(f"  Representations: {len(representations)} layers in {repr_dir}")
    print(f"  Metadata: {metadata_file}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        sys.exit(1)
