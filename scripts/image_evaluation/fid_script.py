#!/usr/bin/env python3

"""
Calculate FID scores between a reference image set and multiple subdirectories.

Example usage:
    python scripts/image_evaluation/fid_script.py \
        --reference_path /mnt/evafs/groups/mi2lab/jcwalina/results/test/reference_images \
        --subdirs_parent_path /mnt/evafs/groups/mi2lab/jcwalina/results/test/generated_sets \
        --output_file /mnt/evafs/groups/mi2lab/jcwalina/results/test/fid_results.txt
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from src.utils.fid import calculate_fid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate FID between reference images and multiple subdirectories."
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        required=True,
        help="Path to reference image directory",
    )
    parser.add_argument(
        "--subdirs_parent_path",
        type=str,
        required=True,
        help="Path to parent directory containing subdirectories with images",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional: Save results to a text file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    reference_path = Path(args.reference_path)
    subdirs_parent_path = Path(args.subdirs_parent_path)

    # Validate paths
    if not reference_path.exists():
        print(f"✗ ERROR: Reference path does not exist: {reference_path}")
        return 1

    if not subdirs_parent_path.exists():
        print(f"✗ ERROR: Subdirectories parent path does not exist: {subdirs_parent_path}")
        return 1

    print("=" * 80)
    print("BATCH FID EVALUATION")
    print("=" * 80)
    print(f"Reference: {reference_path}")
    print(f"Parent directory: {subdirs_parent_path}")
    print("=" * 80)

    # Find all subdirectories containing images
    subdirs = []
    for item in sorted(subdirs_parent_path.iterdir()):
        if item.is_dir():
            # Check if directory contains images
            image_files = (
                list(item.glob("*.png")) + list(item.glob("*.jpg")) + list(item.glob("*.jpeg"))
            )
            if image_files:
                subdirs.append(item)

    if not subdirs:
        print("✗ ERROR: No subdirectories with images found")
        return 1

    print(f"Found {len(subdirs)} subdirectories with images\n")

    # Calculate FID for each subdirectory
    results = []
    try:
        for idx, subdir in enumerate(subdirs, 1):
            print(f"\n[{idx}/{len(subdirs)}] Processing: {subdir.name}")
            print("-" * 80)

            fid_score = calculate_fid(
                images_first_set_path=str(reference_path),
                images_second_set_path=str(subdir),
                verbose=False,
            )

            results.append((subdir.name, fid_score))
            print(f"✓ FID score for {subdir.name}: {fid_score:.4f}")

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Directory':<50} {'FID Score':>15}")
        print("-" * 80)
        for name, score in results:
            print(f"{name:<50} {score:>15.4f}")
        print("=" * 80)

        # Save to file if requested
        if args.output_file:
            output_path = Path(args.output_file)
            with open(output_path, "w") as f:
                f.write(f"Reference: {reference_path}\n")
                f.write(f"Parent: {subdirs_parent_path}\n")
                f.write("=" * 80 + "\n")
                f.write(f"{'Directory':<50} {'FID Score':>15}\n")
                f.write("-" * 80 + "\n")
                for name, score in results:
                    f.write(f"{name:<50} {score:>15.4f}\n")
            print(f"\n✓ Results saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
