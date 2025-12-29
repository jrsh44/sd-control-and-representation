#!/usr/bin/env python3
"""
Merge partial feature sum files from parallel array jobs into a single file.

Example usage:
    python scripts/data/merge_feature_sums.py \
        --input_dir "/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp16_topk32_lr5em5_ep2_bs4096/feature_sums" \
        --pattern "*.pt" \
        --output_path "/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp16_topk32_lr5em5_ep2_bs4096/feature_sums/merged_feature_sums.pt"
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
    """Merge multiple feature sum files, combining across datasets.

    Files with same concept, concept_value but different datasets: add sums and counts
    Files with same dataset, concept, concept_value: keep only one (duplicate)
    """
    if not partial_files:
        raise ValueError("No files to merge")

    print(f"Loading {len(partial_files)} files...")

    # Load all data
    all_data = []
    for file_path in partial_files:
        data = torch.load(file_path, map_location="cpu")
        all_data.append(data)
        print(f"  Loaded {file_path.name}")

    # Group by (concept, concept_value) and track unique datasets
    # Key: (concept, concept_value), Value: list of data dicts
    groups = {}

    for data in all_data:
        key = (data["concept"], data["concept_value"])
        if key not in groups:
            groups[key] = []
        groups[key].append(data)

    print(f"\nFound {len(groups)} unique (concept, concept_value) combinations")

    # Merge each group - indexed by concept_value
    merged_data = {}
    for (concept, concept_value), group_data in groups.items():
        print(f"\nMerging: concept={concept}, concept_value={concept_value}")

        # Remove duplicates (same dataset)
        seen_datasets = {}
        unique_data = []
        for data in group_data:
            dataset_name = data["dataset_name"]
            if dataset_name not in seen_datasets:
                seen_datasets[dataset_name] = data
                unique_data.append(data)
            else:
                print(f"  Skipping duplicate from dataset: {dataset_name}")

        if len(unique_data) == 0:
            continue

        # Create entry for this concept_value
        merged = {
            "timesteps": unique_data[0]["timesteps"],
            "datasets": [unique_data[0]["dataset_name"]],
            "sums_per_timestep": {},
            "counts_per_timestep": {},
        }

        # Initialize with first entry's data
        for timestep, sum_tensor in unique_data[0]["sums_per_timestep"].items():
            merged["sums_per_timestep"][timestep] = sum_tensor.clone()
        for timestep, count in unique_data[0]["counts_per_timestep"].items():
            merged["counts_per_timestep"][timestep] = count

        # Add data from other datasets
        for data in unique_data[1:]:
            dataset_name = data["dataset_name"]
            print(f"  Adding data from dataset: {dataset_name}")
            merged["datasets"].append(dataset_name)

            # Add sums
            for timestep, sum_tensor in data["sums_per_timestep"].items():
                if timestep in merged["sums_per_timestep"]:
                    merged["sums_per_timestep"][timestep] += sum_tensor
                else:
                    merged["sums_per_timestep"][timestep] = sum_tensor.clone()

            # Add counts
            for timestep, count in data["counts_per_timestep"].items():
                if timestep in merged["counts_per_timestep"]:
                    merged["counts_per_timestep"][timestep] += count
                else:
                    merged["counts_per_timestep"][timestep] = count

        print(f"  Merged {len(unique_data)} datasets: {merged['datasets']}")
        merged_data[concept_value] = merged

    # Create "all" group that combines everything across all concepts, concept_values, and datasets
    print(f"\nCreating 'all' group (combining all observations)...")
    all_group = {
        "timesteps": all_data[0]["timesteps"],
        "datasets": [],
        "sums_per_timestep": {},
        "counts_per_timestep": {},
    }

    # Track unique datasets
    seen_datasets_all = set()

    # Combine all data from all files (deduplicating by dataset to avoid double-counting)
    dataset_contributions = {}  # dataset_name -> data dict
    for data in all_data:
        dataset_name = data["dataset_name"]
        if dataset_name not in dataset_contributions:
            dataset_contributions[dataset_name] = data
            seen_datasets_all.add(dataset_name)

    all_group["datasets"] = sorted(list(seen_datasets_all))

    # Sum across all unique datasets
    first = True
    for dataset_name, data in dataset_contributions.items():
        if first:
            # Initialize with first dataset's data
            for timestep, sum_tensor in data["sums_per_timestep"].items():
                all_group["sums_per_timestep"][timestep] = sum_tensor.clone()
            for timestep, count in data["counts_per_timestep"].items():
                all_group["counts_per_timestep"][timestep] = count
            first = False
        else:
            # Add data from other datasets
            for timestep, sum_tensor in data["sums_per_timestep"].items():
                if timestep in all_group["sums_per_timestep"]:
                    all_group["sums_per_timestep"][timestep] += sum_tensor
                else:
                    all_group["sums_per_timestep"][timestep] = sum_tensor.clone()

            for timestep, count in data["counts_per_timestep"].items():
                if timestep in all_group["counts_per_timestep"]:
                    all_group["counts_per_timestep"][timestep] += count
                else:
                    all_group["counts_per_timestep"][timestep] = count

    print(f"  Combined {len(dataset_contributions)} unique datasets: {all_group['datasets']}")

    # Add the "all" group to merged_data dict
    merged_data["all"] = all_group

    return merged_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge partial feature sum files from parallel array jobs"
    )

    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing .pt files to merge"
    )
    parser.add_argument(
        "--pattern", type=str, default="*.pt", help="Glob pattern to match files (default: *.pt)"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path where merged file will be saved"
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Delete input files after successful merge"
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Find input files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"✗ Error: Input directory does not exist: {input_dir}")
        return 1

    if not input_dir.is_dir():
        print(f"✗ Error: Input path is not a directory: {input_dir}")
        return 1

    partial_files = sorted(input_dir.glob(args.pattern))

    if not partial_files:
        print(f"✗ Error: No files found matching pattern '{args.pattern}' in {input_dir}")
        return 1

    print(f"Found {len(partial_files)} files to merge")
    print(f"Output will be saved to: {args.output_path}\n")

    # Merge files
    try:
        merged_data = merge_feature_sums(partial_files)
    except Exception as e:
        print(f"\n✗ Error during merge: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Save merged file
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving merged data to {output_path}...")
    torch.save(merged_data, output_path)

    # Verify saved file
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved successfully ({file_size_mb:.2f} MB)")

    # Print summary
    print(f"\n{'=' * 60}")
    print("MERGE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Input files processed: {len(partial_files)}")
    print(f"Unique concept_value groups: {len(merged_data)}")
    print(f"\nGroups:")
    for concept_value, data in sorted(merged_data.items()):
        print(f"  - concept_value={concept_value}: {len(data['datasets'])} datasets merged")
    print(f"{'=' * 60}")

    # Cleanup if requested
    if args.cleanup:
        print(f"\nCleaning up {len(partial_files)} input files...")
        for file_path in partial_files:
            file_path.unlink()
            print(f"  Deleted {file_path.name}")
        print("✓ Cleanup complete")

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
