#!/usr/bin/env python3
"""
Merge partial feature sum files from parallel array jobs into a single file.

Example usage:
    python scripts/data/merge_feature_sums.py \
        --input_pattern "results/sae_scores/unet_mid_att_canvas_present_job*.pt" \
        --output_path "results/sae_scores/unet_mid_att_canvas_present_feature_sums.pt" \
        --cleanup
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch


def find_partial_files(pattern: str) -> List[Path]:
    """Find all files matching the pattern."""
    pattern_path = Path(pattern)
    parent = pattern_path.parent
    glob_pattern = pattern_path.name

    files = sorted(parent.glob(glob_pattern))
    return files


def merge_feature_sums(partial_files: List[Path]) -> Dict:
    """
    Merge multiple partial feature sum files into one.

    Args:
        partial_files: List of paths to partial .pt files

    Returns:
        Merged dictionary with aggregated sums and counts
    """
    if not partial_files:
        raise ValueError("No partial files found to merge")

    print(f"Found {len(partial_files)} partial files to merge")

    # Initialize merged dictionaries
    merged_sums_true = {}
    merged_counts_true = {}
    merged_sums_false = {}
    merged_counts_false = {}
    all_timesteps = set()

    # Process each partial file
    for i, file_path in enumerate(partial_files, 1):
        print(f"  [{i}/{len(partial_files)}] Loading {file_path.name}...")

        try:
            data = torch.load(file_path, map_location="cpu")
        except Exception as e:
            print(f"    ✗ Failed to load {file_path}: {e}")
            continue

        # Validate structure
        required_keys = [
            "sums_true_per_timestep",
            "counts_true_per_timestep",
            "sums_false_per_timestep",
            "counts_false_per_timestep",
        ]
        if not all(k in data for k in required_keys):
            print(f"    ✗ Skipping {file_path}: missing required keys")
            continue

        # Merge sums_true_per_timestep
        for timestep, tensor in data["sums_true_per_timestep"].items():
            all_timesteps.add(timestep)
            if timestep not in merged_sums_true:
                # Initialize with float64 for precision
                merged_sums_true[timestep] = tensor.clone().to(dtype=torch.float64)
                merged_counts_true[timestep] = 0
            else:
                # Accumulate in float64
                merged_sums_true[timestep] += tensor.to(dtype=torch.float64)

            merged_counts_true[timestep] += data["counts_true_per_timestep"][timestep]

        # Merge sums_false_per_timestep
        for timestep, tensor in data["sums_false_per_timestep"].items():
            all_timesteps.add(timestep)
            if timestep not in merged_sums_false:
                merged_sums_false[timestep] = tensor.clone().to(dtype=torch.float64)
                merged_counts_false[timestep] = 0
            else:
                merged_sums_false[timestep] += tensor.to(dtype=torch.float64)

            merged_counts_false[timestep] += data["counts_false_per_timestep"][timestep]

        print(f"    ✓ Merged (timesteps: {len(data.get('timesteps', []))})")

    # Create final merged structure
    merged_data = {
        "sums_true_per_timestep": merged_sums_true,
        "counts_true_per_timestep": merged_counts_true,
        "sums_false_per_timestep": merged_sums_false,
        "counts_false_per_timestep": merged_counts_false,
        "timesteps": sorted(all_timesteps),
    }

    return merged_data


def print_summary(merged_data: Dict):
    """Print summary statistics of merged data."""
    print("\n" + "=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)

    timesteps = merged_data["timesteps"]
    print(f"Total timesteps: {len(timesteps)}")
    if timesteps:
        print(f"Timestep range: {min(timesteps)} - {max(timesteps)}")

    # Concept TRUE stats
    total_samples_true = sum(merged_data["counts_true_per_timestep"].values())
    print(f"\nConcept=TRUE samples: {total_samples_true:,}")
    if merged_data["sums_true_per_timestep"]:
        example_timestep = timesteps[0]
        num_features = merged_data["sums_true_per_timestep"][example_timestep].shape[0]
        print(f"  Features per timestep: {num_features:,}")
        dtype = merged_data["sums_true_per_timestep"][example_timestep].dtype
        print(f"  Dtype: {dtype}")

    # Concept FALSE stats
    total_samples_false = sum(merged_data["counts_false_per_timestep"].values())
    print(f"\nConcept=FALSE samples: {total_samples_false:,}")
    if merged_data["sums_false_per_timestep"]:
        example_timestep = timesteps[0]
        num_features = merged_data["sums_false_per_timestep"][example_timestep].shape[0]
        print(f"  Features per timestep: {num_features:,}")
        dtype = merged_data["sums_false_per_timestep"][example_timestep].dtype
        print(f"  Dtype: {dtype}")

    # Memory estimate
    if merged_data["sums_true_per_timestep"]:
        bytes_per_tensor = num_features * 8  # float64
        total_tensors = len(timesteps) * 2  # true + false
        total_bytes = bytes_per_tensor * total_tensors
        print(f"\nEstimated size: {total_bytes / (1024**2):.1f} MB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge partial feature sum files from parallel array jobs"
    )
    parser.add_argument(
        "--input_pattern",
        type=str,
        required=True,
        help='Glob pattern for input files (e.g., "results/sae_scores/job*.pt")',
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged output file",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete partial files after successful merge",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without actually merging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 80)
    print("MERGE FEATURE SUMS")
    print("=" * 80)
    print(f"Input pattern: {args.input_pattern}")
    print(f"Output path: {args.output_path}")
    print("-" * 80)

    # Find partial files
    try:
        partial_files = find_partial_files(args.input_pattern)
        if not partial_files:
            print(f"✗ No files found matching pattern: {args.input_pattern}")
            return 1

        print(f"\nFound {len(partial_files)} files:")
        for f in partial_files:
            size_mb = f.stat().st_size / (1024**2)
            print(f"  - {f.name} ({size_mb:.2f} MB)")

        if args.dry_run:
            print("\n[DRY RUN] Would merge these files. Exiting.")
            return 0

    except Exception as e:
        print(f"✗ Error finding files: {e}")
        return 1

    # Merge files
    print("\n" + "=" * 80)
    print("MERGING FILES")
    print("=" * 80)
    try:
        merged_data = merge_feature_sums(partial_files)
        print("✓ Merge complete")

        # Print summary
        print_summary(merged_data)

    except Exception as e:
        print(f"\n✗ Error during merge: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Save merged file
    print("\n" + "=" * 80)
    print("SAVING MERGED FILE")
    print("=" * 80)
    try:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving to {output_path}...")
        torch.save(merged_data, output_path)

        output_size_mb = output_path.stat().st_size / (1024**2)
        print(f"✓ Saved successfully ({output_size_mb:.2f} MB)")

    except Exception as e:
        print(f"✗ Error saving file: {e}")
        return 1

    # Cleanup if requested
    if args.cleanup:
        print("\n" + "=" * 80)
        print("CLEANING UP PARTIAL FILES")
        print("=" * 80)
        for f in partial_files:
            try:
                f.unlink()
                print(f"  ✓ Deleted {f.name}")
            except Exception as e:
                print(f"  ✗ Failed to delete {f.name}: {e}")

    print("\n" + "=" * 80)
    print("✓ MERGE COMPLETE")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
