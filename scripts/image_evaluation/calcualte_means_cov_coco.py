#!/usr/bin/env python3
"""
Script to extract and save Inception v3 statistics from COCO-2017 validation set.

This script loads images from COCO-2017 validation dataset using fiftyone,
extracts features using Inception v3, and saves the mean vector and covariance
matrix for later FID calculations.

Example usage:
    # Extract statistics from 5000 COCO images
    uv run calcualte_means_cov_coco.py \\
        --output-dir ./fid_stats \\
        --num-images 5000

    # Extract statistics from person images only
    uv run scripts/image_evaluation/calcualte_means_cov_coco.py \
        --output-dir ./mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/fid/coco \
        --num-images 100 \
        --classes person

    # Multiple classes
    uv run calcualte_means_cov_coco.py \\
        --output-dir ./fid_stats \\
        --num-images 2000 \\
        --classes person car dog
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.fid import extract_and_save_coco_statistics
from src.utils.wandb import get_system_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save Inception v3 statistics from COCO-2017 validation set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 5000 images
  python calcualte_means_cov_coco.py --output-dir ./fid_stats --num-images 5000
  
  # Filter by class
  python calcualte_means_cov_coco.py --output-dir ./fid_stats --num-images 1000 --classes person
  
  # Multiple classes
  python calcualte_means_cov_coco.py --output-dir ./fid_stats --num-images 2000 --classes person car dog
  
  # Specify device
  python calcualte_means_cov_coco.py --output-dir ./fid_stats --num-images 5000 --device cuda
  
  # Quiet mode
  python calcualte_means_cov_coco.py --output-dir ./fid_stats --num-images 5000 --quiet

Output files will be automatically named based on parameters:
  - coco2017_val_{num_images}_mean.npy
  - coco2017_val_{num_images}_cov.npy
  - coco2017_val_{num_images}_{classes}_mean.npy (if classes specified)
  - coco2017_val_{num_images}_{classes}_cov.npy (if classes specified)
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where statistics will be saved as .npy files",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=5000,
        help="Number of images to load from COCO-2017 validation set (default: 5000)",
    )

    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="List of COCO classes to filter images (e.g., person car dog). "
        "If not specified, uses all images.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use for computation (default: auto-detect)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    parser.add_argument(
        "--skip_wandb",
        action="store_true",
        help="Skip Weights & Biases logging",
    )

    args = parser.parse_args()

    # Validate output directory parent exists (create output_dir itself in the function)
    output_path = Path(args.output_dir)
    if output_path.exists() and not output_path.is_dir():
        print(f"Error: Output path exists but is not a directory: {output_path}", file=sys.stderr)
        sys.exit(1)

    verbose = not args.quiet

    # Determine device for system metrics
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    if not args.skip_wandb:
        wandb.login()
        dataset_name = f"coco2017_val_{args.num_images}"
        if args.classes:
            dataset_name += f"_{'_'.join(args.classes[:3])}"

        wandb.init(
            project="sd-control-representation",
            entity="bartoszjezierski28-warsaw-university-of-technology",
            name=f"fid_stats_{dataset_name}",
            config={
                "output_dir": args.output_dir,
                "num_images": args.num_images,
                "classes": args.classes,
                "device": device,
                "dataset_name": dataset_name,
            },
            tags=["fid", "coco2017", "statistics", "inception_v3"],
        )

        # Log initial system metrics
        system_metrics_start = get_system_metrics(device)
        wandb.log({"start/" + k: v for k, v in system_metrics_start.items()})

    try:
        # Extract and save COCO statistics with timing
        start_time = time.time()

        if not args.skip_wandb:
            wandb.log({"status": "extracting_features"})

        mean_path, cov_path = extract_and_save_coco_statistics(
            output_dir=args.output_dir,
            num_images=args.num_images,
            classes=args.classes,
            device=device,
            verbose=verbose,
        )

        total_time = time.time() - start_time

        if verbose:
            print("\n" + "=" * 80)
            print("SUCCESS")
            print("=" * 80)
            print(f"Mean vector saved to: {mean_path}")
            print(f"Covariance matrix saved to: {cov_path}")
            print(f"Total processing time: {total_time:.2f}s")
            print("\nYou can now use these statistics for efficient FID calculation.")
            print("=" * 80)

        # Log final metrics to wandb
        if not args.skip_wandb:
            system_metrics_end = get_system_metrics(device)
            wandb.log(
                {
                    "final/total_time_seconds": total_time,
                    "final/num_images_processed": args.num_images,
                    "final/time_per_image": total_time / args.num_images,
                    "final/mean_path": str(mean_path),
                    "final/cov_path": str(cov_path),
                }
            )

            # Calculate memory delta if on GPU
            if device == "cuda":
                memory_delta = system_metrics_end.get(
                    "gpu_memory_mb", 0
                ) - system_metrics_start.get("gpu_memory_mb", 0)
                wandb.log({"final/gpu_memory_delta_mb": memory_delta})

            wandb.finish()

    except Exception as e:
        if not args.skip_wandb:
            wandb.log({"error": str(e), "status": "failed"})
            wandb.finish(exit_code=1)

        print(f"\nError during processing: {e}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
