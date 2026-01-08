#!/usr/bin/env python3

"""
Example usage:
    python scripts/image_evaluation/fid_script.py \
        --images_first_set_path /mnt/evafs/groups/mi2lab/jcwalina/results/test/test_image_set_1 \
        --images_second_set_path /mnt/evafs/groups/mi2lab/jcwalina/results/test/test_image_set_2 
"""  # noqa: E501

import argparse
import os
import sys
from os import listdir  # noqa: F401
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from dotenv import load_dotenv
from PIL import Image
from scipy.linalg import sqrtm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate FID between two image datasets.")
    # Dataset parameters
    parser.add_argument(
        "--images_first_set_path",
        type=str,
        required=True,
        help="Path to first set of images for FID evaluation",
    )
    parser.add_argument(
        "--images_second_set_path",
        type=str,
        required=True,
        help="Path to second set of images for FID evaluation",
    )
    # skip wandb
    parser.add_argument("--skip-wandb", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("EVALUATING FID SCORES")
    print("=" * 80)
    print(f"Image path: {args.images_first_set_path}")
    print(f"Image path: {args.images_second_set_path}")
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

        FID_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)  # noqa: N806
        print("✓ FID model loaded successfully.")
        FID_model.fc = torch.nn.Identity()
        FID_model.eval()
        FID_model.to(device)

        # Inception v3 expects 299x299 images with ImageNet normalization
        preprocess = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Count images first
        num_images_1 = len(
            [
                f
                for f in os.listdir(args.images_first_set_path)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        image_first_set_vectors = torch.empty((num_images_1, 2048), device=device)
        image_files_1 = [
            f
            for f in os.listdir(args.images_first_set_path)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        for idx, images in enumerate(image_files_1):
            print(f"Processing image {idx + 1}/{len(image_files_1)}: {images}")
            image_path = os.path.join(args.images_first_set_path, images)
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = FID_model(image_tensor)
            image_first_set_vectors[idx] = features
        print(f"Processed {num_images_1} images from first set.")

        # Count images second
        num_images_2 = len(
            [
                f
                for f in os.listdir(args.images_second_set_path)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        image_second_set_vectors = torch.empty((num_images_2, 2048), device=device)
        image_files_2 = [
            f
            for f in os.listdir(args.images_second_set_path)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        for idx, images in enumerate(image_files_2):
            print(f"Processing image {idx + 1}/{len(image_files_2)}: {images}")
            image_path = os.path.join(args.images_second_set_path, images)
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = FID_model(image_tensor)
            image_second_set_vectors[idx] = features
        print(f"Processed {num_images_2} images from second set.")

        mu1 = np.mean(image_first_set_vectors.cpu().numpy(), axis=0)
        sigma1 = np.cov(image_first_set_vectors.cpu().numpy(), rowvar=False)
        mu2 = np.mean(image_second_set_vectors.cpu().numpy(), axis=0)
        sigma2 = np.cov(image_second_set_vectors.cpu().numpy(), rowvar=False)

        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid_score = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        print(f"\nFID score between the two image sets: {fid_score}")

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
