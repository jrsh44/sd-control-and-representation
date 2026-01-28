"""
Script to extract and save Inception v3 statistics (mean and covariance) from images.

This script processes subdirectories containing images, extracts features using Inception v3,
and saves the mean vector and covariance matrix for later FID calculations.

Usage:
    # Process all subdirectories in a parent directory
    uv run scripts/image_evaluation/calculate_means_cov.py \
        --parent-dir path/to/images \
        --output-dir path/to/output

    # With options
    uv run scripts/image_evaluation/calculate_means_cov.py \
        --parent-dir path/to/images \
        --output-dir path/to/stats \
        --max-images 1000 \\
        --device cuda \\
        --skip-existing
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.fid import extract_and_save_statistics  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save Inception v3 statistics from subdirectories for FID calculation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subdirectories
  python calculate_means_cov.py --parent-dir ./results/experiments --output-dir ./fid_stats

  # Limit images per subdirectory
  python calculate_means_cov.py --parent-dir ./results --output-dir ./stats --max-images 500

  # Specify device
  python calculate_means_cov.py --parent-dir ./results --output-dir ./stats --device cuda

  # Skip existing statistics
  python calculate_means_cov.py --parent-dir ./results --output-dir ./stats --skip-existing

  # Quiet mode
  python calculate_means_cov.py --parent-dir ./results --output-dir ./stats --quiet
        """,
    )

    parser.add_argument(
        "--parent-dir",
        type=str,
        required=False,
        help="Path to parent directory containing subdirectories with images",
    )

    parser.add_argument(
        "--images-path",
        type=str,
        required=False,
        help="Path to single directory containing images (alternative to --parent-dir)",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        required=False,
        help="Name for the dataset when using --images-path (used in output filenames)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where statistics will be saved as .npy files",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process per subdirectory (default: process all images)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use for computation (default: auto-detect)",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subdirectories that already have statistics computed",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    # Validate that either parent-dir or images-path is provided
    if not args.parent_dir and not args.images_path:
        print("Error: Either --parent-dir or --images-path must be specified", file=sys.stderr)
        sys.exit(1)

    if args.parent_dir and args.images_path:
        print("Error: Cannot specify both --parent-dir and --images-path", file=sys.stderr)
        sys.exit(1)

    if args.images_path and not args.dataset_name:
        print("Error: --dataset-name is required when using --images-path", file=sys.stderr)
        sys.exit(1)

    verbose = not args.quiet

    # Mode 1: Single directory mode
    if args.images_path:
        images_path = Path(args.images_path)
        if not images_path.exists():
            print(f"Error: Images path does not exist: {images_path}", file=sys.stderr)
            sys.exit(1)

        if not images_path.is_dir():
            print(f"Error: Images path is not a directory: {images_path}", file=sys.stderr)
            sys.exit(1)

        # Check if there are any images
        image_files = (
            list(images_path.glob("*.png"))
            + list(images_path.glob("*.jpg"))
            + list(images_path.glob("*.jpeg"))
        )
        if not image_files:
            print(f"Error: No image files found in {images_path}", file=sys.stderr)
            sys.exit(1)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Extract and save statistics
            mean_path, cov_path = extract_and_save_statistics(
                images_path=str(args.images_path),
                output_dir=str(args.output_dir),
                dataset_name=args.dataset_name,
                max_images=args.max_images,
                device=args.device,
                verbose=verbose,
            )

            if verbose:
                print("\n" + "=" * 80)
                print("SUCCESS")
                print("=" * 80)
                print(f"Mean vector saved to: {mean_path}")
                print(f"Covariance matrix saved to: {cov_path}")
                print("=" * 80)

            sys.exit(0)

        except Exception as e:
            print(f"\nError during processing: {e}", file=sys.stderr)
            if verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)

    # Mode 2: Parent directory mode (batch processing)
    parent_dir = Path(args.parent_dir)
    if not parent_dir.exists():
        print(f"Error: Parent directory does not exist: {parent_dir}", file=sys.stderr)
        sys.exit(1)

    if not parent_dir.is_dir():
        print(f"Error: Parent path is not a directory: {parent_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    # Find all subdirectories with images
    subdirs = []
    for item in parent_dir.iterdir():
        if item.is_dir():
            # Check if subdirectory contains images
            image_files = (
                list(item.glob("*.png")) + list(item.glob("*.jpg")) + list(item.glob("*.jpeg"))
            )
            if image_files:
                subdirs.append(item)

    if not subdirs:
        print(f"Error: No subdirectories with images found in {parent_dir}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print("=" * 80)
        print("BATCH STATISTICS EXTRACTION")
        print("=" * 80)
        print(f"Parent directory: {parent_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Found {len(subdirs)} subdirectories with images")
        if args.max_images:
            print(f"Max images per subdirectory: {args.max_images}")
        print("=" * 80)
        print()

    # Process each subdirectory
    total_processed = 0
    total_skipped = 0
    failed = []

    for idx, subdir in enumerate(subdirs, 1):
        dataset_name = subdir.name
        mean_path = output_dir / dataset_name / f"{dataset_name}_mean.npy"
        cov_path = output_dir / dataset_name / f"{dataset_name}_cov.npy"

        # Check if already exists
        if args.skip_existing and mean_path.exists() and cov_path.exists():
            if verbose:
                print(f"[{idx}/{len(subdirs)}] ⏭️  Skipping '{dataset_name}' (already exists)")
            total_skipped += 1
            continue

        if verbose:
            print(f"[{idx}/{len(subdirs)}] 🔄 Processing '{dataset_name}'...")

        try:
            # Extract and save statistics
            mean_path_result, cov_path_result = extract_and_save_statistics(
                images_path=str(subdir),
                output_dir=str(output_dir / dataset_name),
                dataset_name=dataset_name,
                max_images=args.max_images,
                device=args.device,
                verbose=False,  # Suppress verbose output for cleaner batch processing
            )

            if verbose:
                print(f"    ✅ Success: {dataset_name}")
                print(f"       Mean: {mean_path_result}")
                print(f"       Cov:  {cov_path_result}")
                print()

            total_processed += 1

        except Exception as e:
            failed.append((dataset_name, str(e)))
            if verbose:
                print(f"    ❌ Failed: {e}")
                print()

    # Print summary
    if verbose:
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total subdirectories: {len(subdirs)}")
        print(f"  ✅ Processed: {total_processed}")
        print(f"  ⏭️  Skipped: {total_skipped}")
        print(f"  ❌ Failed: {len(failed)}")

        if failed:
            print()
            print("Failed subdirectories:")
            for name, error in failed:
                print(f"  - {name}: {error}")

        print(f"\nOutput directory: {output_dir}")
        print("=" * 80)

    # Exit with error if any failed
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
