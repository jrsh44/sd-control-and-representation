#!/usr/bin/env python3
""""
Example usage:
    uv run scripts/image_evaluation/nudity_unlearning_evaluation.py \
        --image_path /mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation/data/test_images/honda_civic.png \
        --detection_scores_path /mnt/evafs/groups/mi2lab/jcwalina/results/nudity_detection_results.npy
"""  # noqa: E501

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
        "--image_path",
        type=str,
        required=True,
        help="Path to image or directory of images to evaluate unlearning on",
    )
    # feature score path
    parser.add_argument(
        "--detection_scores_path",
        type=str,
        required=True,
        help="Path to .save file for detection scores (numpy array)",
    )
    # skip wandb
    parser.add_argument("--skip-wandb", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("EVALUATING NUDE DETECTION SCORES")
    print("=" * 80)
    print(f"Image path: {args.image_path}")
    print(f"Detection scores path: {args.detection_scores_path}")
    print("-" * 80)

    try:
        # Initialize wandb
        # if not args.skip_wandb:
        #     wandb.login()
        #     wandb.init(
        #         project="sd-control-representation",
        #         entity="bartoszjezierski28-warsaw-university-of-technology",
        #         name="evaluation of nudity",
        #         config={
        #             "image_path": args.image_path,
        #             "detection_scores_path": args.detection_scores_path,
        #         },
        #         tags=["evaluation", "nudity", "detection"],
        #         notes="Feature selection using pretrained SAE",
        #     )

        print("DETECTING NUDITY IN IMAGES")
        detections = detect_images(args.image_path)  # dictionary

        # Save scores
        detection_scores_path = Path(args.detection_scores_path)
        detection_scores_path.parent.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print("SAVING DETECTION SCORES")
        print("=" * 80)
        print(f"Saving detection scores to {detection_scores_path}...")
        torch.save(detections, detection_scores_path)
        print("✓ Detection scores saved successfully.")

        print("\n" + "=" * 80)
        print("Done.")
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
