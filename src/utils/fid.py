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


def calculate_fid(
    images_first_set_path: str,
    images_second_set_path: str,
    device: str = None,
    verbose: bool = True,
) -> float:
    """
    Calculate FID (Fréchet Inception Distance) between two image sets.

    Args:
        images_first_set_path: Path to first set of images
        images_second_set_path: Path to second set of images
        device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        verbose: Whether to print progress messages

    Returns:
        FID score (float)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print("=" * 80)
        print("CALCULATING FID SCORE")
        print("=" * 80)
        print(f"First set: {images_first_set_path}")
        print(f"Second set: {images_second_set_path}")
        print("-" * 80)

    # Load Inception v3 model
    FID_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)  # noqa: N806
    if verbose:
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

    # Process first image set
    image_files_1 = [
        f for f in os.listdir(images_first_set_path) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    num_images_1 = len(image_files_1)
    image_first_set_vectors = torch.empty((num_images_1, 2048), device=device)

    for idx, images in enumerate(image_files_1):
        if verbose:
            print(f"Processing first set: {idx + 1}/{num_images_1}: {images}")
        image_path = os.path.join(images_first_set_path, images)
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = FID_model(image_tensor)
        image_first_set_vectors[idx] = features

    if verbose:
        print(f"✓ Processed {num_images_1} images from first set.")

    # Process second image set
    image_files_2 = [
        f for f in os.listdir(images_second_set_path) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    num_images_2 = len(image_files_2)
    image_second_set_vectors = torch.empty((num_images_2, 2048), device=device)

    for idx, images in enumerate(image_files_2):
        if verbose:
            print(f"Processing second set: {idx + 1}/{num_images_2}: {images}")
        image_path = os.path.join(images_second_set_path, images)
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = FID_model(image_tensor)
        image_second_set_vectors[idx] = features

    if verbose:
        print(f"✓ Processed {num_images_2} images from second set.")

    # Calculate FID
    mu1 = np.mean(image_first_set_vectors.cpu().numpy(), axis=0)
    sigma1 = np.cov(image_first_set_vectors.cpu().numpy(), rowvar=False)
    mu2 = np.mean(image_second_set_vectors.cpu().numpy(), axis=0)
    sigma2 = np.cov(image_second_set_vectors.cpu().numpy(), rowvar=False)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_score = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

    if verbose:
        print(f"\n✓ FID score: {fid_score:.4f}")
        print("=" * 80)

    return fid_score
