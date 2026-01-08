"""
Example usage:
uv run scripts/tests/test_nudenet_wrapper.py \
    --image_path /mnt/evafs/groups/mi2lab/bjezierski/sd-control-and-representation/data/wandb/images/Cache_nudity_sd_v1_5_2layers/didjhxb9/media/images/generated_image_59_3d4de63b9b6d9e8fda61.png \
    --model 320n \
    --threshold 0.7
"""  # noqa: E501

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.nudenet import NudeNetWrapper  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Test NudeNet wrapper for scoring images")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the image to test",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["320n", "640m"],
        default="320n",
        help="Which model to use (default: 320n)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Nudity threshold (default: 0.5)",
    )

    args = parser.parse_args()

    # Validate image path
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    # Initialize wrapper (uses default models)
    print(f"Initializing NudeNet {args.model} model...")
    wrapper = NudeNetWrapper(model_name=args.model)

    # Score image
    print(f"\nScoring image: {image_path}")
    score = wrapper.score_image(str(image_path))
    is_nude = wrapper.is_nude(str(image_path), threshold=args.threshold)

    # Display results
    print("=" * 60)
    print(f"Nudity Score: {score:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Is Nude: {is_nude}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
