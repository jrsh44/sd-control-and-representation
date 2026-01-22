#!/usr/bin/env python3
"""
Merge partial feature sum files from parallel array jobs into a single file.

Example usage:
    python scripts/data/merge_feature_sums.py \
        --input_dir "/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/feature_sums" \
        --pattern "*.pt" \
        --output_path "/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/feature_merged/merged_feature_sums.pt"
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
    """Merge multiple feature sum files.

    Returns a dict where:
    - Keys are concept_value strings (plus 'all' for total)
    - Values contain sums_per_timestep and counts_per_timestep

    Merging rules:
    - Files with same concept_value: add sums and counts
    - Duplicates (same concept, concept_value, dataset_name): keep only one
    - 'all': sum across all concept_values (respecting deduplication)
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

    # Deduplicate: keep only one file per (concept, concept_value, dataset_name)
    # Key: (concept, concept_value, dataset_name) -> data dict
    unique_data = {}
    duplicates_count = 0

    for data in all_data:
        key = (data["concept"], data["concept_value"], data["dataset_name"])
        if key not in unique_data:
            unique_data[key] = data
        else:
            duplicates_count += 1
            print(
                f"  Skipping duplicate: concept={data['concept']}, "
                f"concept_value={data['concept_value']}, dataset={data['dataset_name']}"
            )

    print(
        f"\nAfter deduplication: {len(unique_data)} unique entries ({duplicates_count} duplicates removed)"
    )

    # Group by concept_value
    by_concept_value = {}
    for (concept, concept_value, dataset_name), data in unique_data.items():
        if concept_value not in by_concept_value:
            by_concept_value[concept_value] = []
        by_concept_value[concept_value].append(data)

    print(f"Found {len(by_concept_value)} unique concept_values")

    # Merge each concept_value group
    result = {}

    for concept_value, data_list in by_concept_value.items():
        print(f"\nMerging concept_value='{concept_value}' ({len(data_list)} entries)")

        merged = {"sums_per_timestep": {}, "counts_per_timestep": {}}

        # Sum all entries for this concept_value
        for data in data_list:
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

        result[concept_value] = merged
        print(
            f"  Total counts: {list(merged['counts_per_timestep'].values())[0] if merged['counts_per_timestep'] else 0}"
        )

    # Create 'all' entry - sum across all unique data (already deduplicated)
    print(f"\nCreating 'all' entry (sum across all concept_values)...")

    all_merged = {"sums_per_timestep": {}, "counts_per_timestep": {}}

    for (concept, concept_value, dataset_name), data in unique_data.items():
        # Add sums
        for timestep, sum_tensor in data["sums_per_timestep"].items():
            if timestep in all_merged["sums_per_timestep"]:
                all_merged["sums_per_timestep"][timestep] += sum_tensor
            else:
                all_merged["sums_per_timestep"][timestep] = sum_tensor.clone()

        # Add counts
        for timestep, count in data["counts_per_timestep"].items():
            if timestep in all_merged["counts_per_timestep"]:
                all_merged["counts_per_timestep"][timestep] += count
            else:
                all_merged["counts_per_timestep"][timestep] = count

    result["all"] = all_merged
    print(
        f"  Total counts: {list(all_merged['counts_per_timestep'].values())[0] if all_merged['counts_per_timestep'] else 0}"
    )

    return result


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
