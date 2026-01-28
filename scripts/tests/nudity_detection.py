"""
Detect nudity in images using NudeNet.

This script runs nudity detection on images and saves detection scores.

Usage:
    uv run scripts/tests/nudity_detection.py \
        --images_path /path/to/images \
        --detection_scores_path /path/to/output/scores.npy
"""

import argparse
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")


from src.utils.NudeNet_detector import detect_images  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAE on representations from SD layer using RepresentationDataset."
    )
    # Dataset parameters
    parser.add_argument(
        "--images_path",
        type=str,
        required=True,
        help="Path to training dataset directory (e.g., results/sd_1_5/unet_mid_att)",
    )
    # feature score path
    parser.add_argument(
        "--scores_path",
        type=str,
        required=True,
        help="Path to .npy file for feature scores (load if exists, create and save if not)",
    )
    parser.add_argument("--skip-wandb", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = "/mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation/data/test_images/honda_civic.png"

    print("=" * 80)
    print("EVALUATING NUDE DETECTION SCORES")
    print("=" * 80)
    print(f"Image/s path: {args.images_path}")
    print(f"Scores path: {args.scores_path}")
    print("-" * 80)

    try:
        print("DETECTING NUDITY IN IMAGES")
        detections = detect_images(args.images_path)

        for img_path, detection in detections.items():
            print(f"Image: {img_path}")
            print(f"Detection: {detection}")

        # results_path = "/mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation/results/nudity_detection_results.txt"
        results_path = args.scores_path

        # Save results to a text file, create directories if they don't exist
        results_path_parent = Path(results_path).parent
        results_path_parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w") as f:
            for img_path, detection in detections.items():
                f.write(f"Image: {img_path}\n")
                f.write(f"Detection: {detection}\n")
                f.write("\n")

        print(f"✓ Detection results saved to {results_path}")

        return 0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
