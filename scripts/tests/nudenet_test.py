"""
Example usage:
uv run scripts/tests/nudenet_test.py \
    --image_path /mnt/evafs/groups/mi2lab/bjezierski/sd-control-and-representation/data/wandb/images/Cache_nudity_sd_v1_5_2layers/didjhxb9/media/images/generated_image_59_3d4de63b9b6d9e8fda61.png \
    --output_dir /mnt/evafs/groups/mi2lab/bjezierski/results/nudenet_test
"""  # noqa: E501

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import onnxruntime as ort
from dotenv import load_dotenv
from nudenet import NudeDetector
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

# Part of the code that makes nudenet work (onnx fix)
# ----------------------
num_cpus = len(os.sched_getaffinity(0))
_original_init = ort.InferenceSession.__init__


def _patched_init(self, model_path, sess_options=None, providers=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = num_cpus
    sess_options.inter_op_num_threads = 1
    _original_init(self, model_path, sess_options=sess_options, providers=providers, **kwargs)


ort.InferenceSession.__init__ = _patched_init
# -----------------------


def main():
    parser = argparse.ArgumentParser(description="Test NudeNet models on an image")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the image to test",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (default: same as image directory)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["320n", "640m"],
        default=["320n", "640m"],
        help="Which models to test (default: both)",
    )

    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = image_path.parent

    if "640m" in args.models:
        model_640m_path = Path(os.environ.get("CACHE_DIR", "")) / "models" / "nudenet" / "640m.onnx"
    else:
        model_640m_path = None

    original_image = Image.open(image_path)

    models_to_test = []
    if "320n" in args.models:
        models_to_test.append(("320n", None, 320))
    if "640m" in args.models:
        models_to_test.append(("640m", str(model_640m_path), 640))

    results = []
    for model_name, model_path, resolution in models_to_test:
        print(f"\n{'=' * 60}")
        print(f"Testing {model_name} model")
        print(f"{'=' * 60}")

        detector = NudeDetector(
            model_path=model_path,
            inference_resolution=resolution,
        )
        detections = detector.detect(str(image_path))
        print(f"Detections: {detections}")

        # Censor image
        censored_filename = f"{image_path.stem}_censored_{model_name}{image_path.suffix}"
        censored_path = output_dir / censored_filename
        detector.censor(str(image_path), output_path=str(censored_path))
        censored_image = Image.open(censored_path)

        results.append((model_name, detections, censored_image))

    # Display comparison
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(6 * (len(results) + 1), 6))

    # Handle single subplot case
    if len(results) == 0:
        axes = [axes]

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for idx, (model_name, detections, censored_img) in enumerate(results, 1):
        axes[idx].imshow(censored_img)
        axes[idx].set_title(f"{model_name}\n({len(detections)} detections)")
        axes[idx].axis("off")

    plt.tight_layout()

    # Save comparison
    comparison_filename = f"{image_path.stem}_nudenet_comparison.png"
    comparison_path = output_dir / comparison_filename
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")

    print("\n" + "=" * 60)
    print(f"Comparison saved to: {comparison_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
