#!/usr/bin/env python3
"""
Generate single representation using SD pipeline and save to .pt file.

Usage:
    uv run scripts/tests/create_representation.py \
        --prompt "a cat" \
        --layer UNET_MID_ATT \
        --output test_data/real_representation.pt \
        --steps 50
"""

import argparse
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from src.models.config import ModelRegistry  # noqa: E402
from src.models.sd_v1_5 import LayerPath, capture_layer_representations  # noqa: E402
from src.utils.model_loader import ModelLoader  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate one real representation")
    parser.add_argument("--prompt", type=str, default="a cat", help="Text prompt")
    parser.add_argument(
        "--layer",
        type=str,
        default="UNET_MID_ATT",
        help="Layer name (e.g., UNET_MID_ATT, TEXT_EMBEDDING_FINAL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_data/real_representation.pt",
        help="Output .pt file path",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Parse layer
    try:
        layer = LayerPath[args.layer.upper()]
    except KeyError:
        print(f"ERROR: Unknown layer '{args.layer}'")
        print(f"Available layers: {', '.join([layer.name for layer in LayerPath])}")
        return 1

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Generating representation for: {args.prompt}")
    print(f"Layer: {layer.name}")
    print(f"Steps: {args.steps}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model_registry = ModelRegistry.FINETUNED_SAEURON
    loader = ModelLoader(model_enum=model_registry)
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    pipe = loader.load_model(device=device)
    print(f"✅ Model loaded on {device}")

    # Generate
    print(f"\nGenerating with prompt: '{args.prompt}'...")
    generator = torch.Generator(device).manual_seed(args.seed)

    representations = capture_layer_representations(
        pipe=pipe,
        prompt=args.prompt,
        layer_paths=[layer],
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    representation = representations[0][3]
    print(f"✅ Generated representation shape: {representation.shape}")

    # Save
    torch.save(representation, output_path)
    print(f"✅ Saved to: {output_path.absolute()}")

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print(f"\nRepresentation shape: {representation.shape}")
    print(f"File: {output_path.absolute()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
