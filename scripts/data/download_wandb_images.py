"""Download images from specific wandb runs based on CSV filter.
Usage:

uv run scripts/data/download_wandb_images.py \
    --csv-file /mnt/evafs/groups/mi2lab/bjezierski/sd-control-and-representation/data/wandb/runs07_12_2025.csv \
    --entity "bartoszjezierski28-warsaw-university-of-technology" \
    --project "sd-control-representation" \
    --run-names "Cache_cc3m-wds_sd_v1_5_2layers" "Cache_nudity_sd_v1_5_2layers" \
    --state finished \
    --output-dir /mnt/evafs/groups/mi2lab/bjezierski/sd-control-and-representation/data/wandb/images
"""  # noqa: E501

import argparse
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")
from src.utils.wandb import download_images_from_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Download images from wandb runs filtered by name and state"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        required=True,
        help="Path to CSV file with run data (from export_wandb_runs.py)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        required=True,
        help="wandb entity (username or team)",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="wandb project name",
    )
    parser.add_argument(
        "--run-names",
        type=str,
        nargs="+",
        default=["Cache_cc3m-wds_sd_v1_5_2layers", "Cache_nudity_sd_v1_5_2layers"],
        help="List of run names to filter (default: Cache_cc3m-wds_sd_v1_5_2layers Cache_nudity_sd_v1_5_2layers)",  # noqa: E501
    )
    parser.add_argument(
        "--state",
        type=str,
        default="finished",
        help="Run state to filter (default: finished)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for images (default: data/wandb/images/<run_name>)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log progress every N downloaded files (default: 50)",
    )

    args = parser.parse_args()

    # Read CSV file
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return 1

    print(f"Reading runs from: {csv_path}")
    print(f"Filtering for run names: {args.run_names}")
    print(f"Filtering for state: {args.state}")
    print("=" * 80)

    # Parse CSV and filter runs
    matching_runs = []
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["run_name"] in args.run_names and row["state"] == args.state:
                matching_runs.append(row)

    print(f"Found {len(matching_runs)} matching runs")

    if not matching_runs:
        print("No matching runs found. Exiting.")
        return 0

    # Download images from each matching run
    total_images = 0
    skipped_runs = 0
    for i, run_data in enumerate(matching_runs, 1):
        run_id = run_data["run_id"]
        run_name = run_data["run_name"]

        print(f"\n[{i}/{len(matching_runs)}] Processing run: {run_name} (ID: {run_id})")

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir) / run_name / run_id
        else:
            output_dir = (
                Path(__file__).parent.parent.parent
                / "data"
                / "wandb"
                / "images"
                / run_name
                / run_id
            )

        # Check if directory already exists
        if output_dir.exists():
            print("  ⏭️  Skipping - directory already exists")
            skipped_runs += 1
            continue

        print(f"  Output directory: {output_dir}")

        try:
            downloaded = download_images_from_run(
                entity=args.entity,
                project=args.project,
                run_id=run_id,
                output_dir=str(output_dir),
                log_every=args.log_every,
            )

            if downloaded:
                print(f"  ✅ Downloaded {len(downloaded)} images")
                total_images += len(downloaded)
            else:
                print("  ℹ️  No images found in this run")

        except Exception as e:
            print(f"  ❌ ERROR downloading from run {run_id}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"SUMMARY: Downloaded {total_images} images from {len(matching_runs)} runs")
    print(f"  Skipped: {skipped_runs} runs (already downloaded)")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
