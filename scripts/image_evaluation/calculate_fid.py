#!/usr/bin/env python3
"""
Script to calculate FID scores from pre-computed statistics files.

This script compares intervention subdirectories against COCO-2017 reference statistics.

Usage:
    python scripts/image_evaluation/calculate_fid.py \
        --coco-dir /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/fid/coco \
        --parent-dir /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/fid/nudity/exposed_feet \
        --output /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/fid/scores/exposed_feet_fid.csv

Directory structure expected:
    parent-dir/
        fn08_if12.8/
            *_mean.npy
            *_cov.npy
        fn05_if-3.2/
            *_mean.npy
            *_cov.npy
        no_intervention/
            *_mean.npy
            *_cov.npy
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.fid import calculate_fid_from_statistics  # noqa: E402


def parse_directory_name(dir_name: str) -> tuple[str, str]:
    """
    Parse directory name to extract feature_number and influence_factor.

    Args:
        dir_name: Directory name like "fn08_if12.8", "fn05_if-3.2", or "no_intervention"

    Returns:
        Tuple of (feature_number, influence_factor) as strings
    """
    if dir_name == "no_intervention":
        return "no_intervention", "no_intervention"

    # Pattern: fn[number]_if[float]
    # Examples: fn08_if12.8, fn05_if-3.2, fn10_if0.5
    pattern = r"fn(\d+)_if([-+]?\d*\.?\d+)"
    match = re.match(pattern, dir_name)

    if match:
        feature_number = match.group(1)
        influence_factor = match.group(2)
        return feature_number, influence_factor
    else:
        raise ValueError(
            f"Cannot parse directory name: {dir_name}. Expected format: fn[nr]_if[float] or no_intervention"
        )


def find_statistics_files(directory: Path) -> tuple[Path, Path]:
    """
    Find mean and cov files in a directory.

    Args:
        directory: Path to directory containing statistics files

    Returns:
        Tuple of (mean_path, cov_path)

    Raises:
        FileNotFoundError: If files are not found
    """
    mean_files = list(directory.glob("*_mean.npy"))
    cov_files = list(directory.glob("*_cov.npy"))

    if not mean_files:
        raise FileNotFoundError(f"No *_mean.npy file found in {directory}")
    if not cov_files:
        raise FileNotFoundError(f"No *_cov.npy file found in {directory}")

    if len(mean_files) > 1:
        print(f"Warning: Multiple mean files found in {directory}, using first one")
    if len(cov_files) > 1:
        print(f"Warning: Multiple cov files found in {directory}, using first one")

    return mean_files[0], cov_files[0]


def main():
    parser = argparse.ArgumentParser(
        description="Calculate FID scores for intervention experiments against COCO-2017",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calculate_fid.py \\
      --coco-dir /path/to/coco_stats \\
      --parent-dir /path/to/nudity/exposed_feet \\
      --output results.csv \\
      --per-timestep
  
  python calculate_fid.py \\
      --coco-dir ./coco \\
      --parent-dir ./interventions/concept_1 \\
      --output fid_scores.csv
        """,
    )

    parser.add_argument(
        "--coco-dir",
        type=str,
        required=True,
        help="Directory containing COCO-2017 reference statistics (*_mean.npy and *_cov.npy)",
    )

    parser.add_argument(
        "--parent-dir",
        type=str,
        required=True,
        help="Parent directory containing subdirectories with intervention statistics (e.g., fn08_if12.8, no_intervention)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file to save results",
    )

    parser.add_argument(
        "--per-timestep",
        action="store_true",
        help="Set per_timestep flag to True in output (default: False)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce verbosity",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Validate COCO directory
    coco_dir = Path(args.coco_dir)
    if not coco_dir.exists():
        print(f"Error: COCO directory not found: {coco_dir}", file=sys.stderr)
        return 1

    # Find COCO statistics files
    try:
        coco_mean_path, coco_cov_path = find_statistics_files(coco_dir)
        if verbose:
            print(f"COCO reference statistics:")
            print(f"  Mean: {coco_mean_path}")
            print(f"  Cov:  {coco_cov_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Validate parent directory
    parent_dir = Path(args.parent_dir)
    if not parent_dir.exists():
        print(f"Error: Parent directory not found: {parent_dir}", file=sys.stderr)
        return 1

    # Extract concept name (last directory in parent path)
    concept = parent_dir.name

    # Find all subdirectories
    subdirs = [d for d in parent_dir.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"Error: No subdirectories found in {parent_dir}", file=sys.stderr)
        return 1

    if verbose:
        print(f"\nConcept: {concept}")
        print(f"Per-timestep: {args.per_timestep}")
        print(f"\nFound {len(subdirs)} subdirectory(ies) to process:")
        for d in sorted(subdirs):
            print(f"  - {d.name}")
        print()

    # Calculate FID for each subdirectory
    results = []
    print("=" * 80)
    print("CALCULATING FID SCORES")
    print("=" * 80)

    for idx, subdir in enumerate(sorted(subdirs), 1):
        dir_name = subdir.name

        if verbose:
            print(f"\n[{idx}/{len(subdirs)}] Processing: {dir_name}")

        try:
            # Parse directory name
            feature_number, influence_factor = parse_directory_name(dir_name)

            # Find statistics files
            exp_mean_path, exp_cov_path = find_statistics_files(subdir)

            if verbose:
                print(f"  Feature number: {feature_number}")
                print(f"  Influence factor: {influence_factor}")
                print(f"  Mean: {exp_mean_path.name}")
                print(f"  Cov:  {exp_cov_path.name}")

            # Calculate FID
            fid_score = calculate_fid_from_statistics(
                mean_path_1=str(exp_mean_path),
                cov_path_1=str(exp_cov_path),
                mean_path_2=str(coco_mean_path),
                cov_path_2=str(coco_cov_path),
                verbose=False,
            )

            results.append(
                {
                    "feature_number": feature_number,
                    "influence_factor": influence_factor,
                    "fid_score": fid_score,
                    "concept": concept,
                    "per_timestep": args.per_timestep,
                }
            )

            if verbose:
                print(f"  ✓ FID Score: {fid_score:.4f}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Still add entry with None for FID score
            try:
                feature_number, influence_factor = parse_directory_name(dir_name)
            except ValueError:
                feature_number, influence_factor = "error", "error"

            results.append(
                {
                    "feature_number": feature_number,
                    "influence_factor": influence_factor,
                    "fid_score": None,
                    "concept": concept,
                    "per_timestep": args.per_timestep,
                    "error": str(e),
                }
            )

    # Create results DataFrame
    df = pd.DataFrame(results)

    # Sort by feature_number and influence_factor
    # Convert to numeric for proper sorting (except 'no_intervention')
    def sort_key(row):
        if row["feature_number"] == "no_intervention":
            return (-1, -1)  # Sort no_intervention first
        try:
            return (int(row["feature_number"]), float(row["influence_factor"]))
        except (ValueError, TypeError):
            return (999999, 999999)  # Sort errors last

    df_sorted = df.iloc[df.apply(sort_key, axis=1).argsort()]

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nConcept: {concept}")
    print(f"Reference: COCO-2017")
    print(f"Per-timestep: {args.per_timestep}")
    print(f"Experiments processed: {len(subdirs)}")
    print(f"Successful: {df['fid_score'].notna().sum()}")
    print(f"Failed: {df['fid_score'].isna().sum()}")
    print()

    # Print table
    print(df_sorted.to_string(index=False))
    print()

    # Save to CSV
    output_path = Path(args.output)

    # Ensure output has .csv extension
    if output_path.suffix != ".csv":
        output_path = output_path.with_suffix(".csv")
        print(f"Note: Added .csv extension to output file: {output_path}")

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_sorted.to_csv(output_path, index=False)
    print(f"✓ Results saved to: {output_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
