import os
import sys
from pathlib import Path

import fiftyone as fo
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


def _load_inception_model(device: str, verbose: bool = False):
    """Load and prepare Inception v3 model for feature extraction."""
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    if verbose:
        print("✓ Inception v3 model loaded successfully.")
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def _get_inception_preprocessing():
    """Get preprocessing transforms for Inception v3."""
    return transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _extract_features_from_directory(
    images_path: str,
    model,
    preprocess,
    device: str,
    max_images: int = None,
    verbose: bool = True,
) -> np.ndarray:
    """Extract Inception features from all images in a directory.

    Args:
        images_path: Path to directory containing images
        model: Inception model for feature extraction
        preprocess: Preprocessing transforms
        device: Device to use for computation
        max_images: Maximum number of images to process (None = all)
        verbose: Whether to print progress

    Returns:
        numpy array of shape (num_images, 2048) containing features
    """
    image_files = [f for f in os.listdir(images_path) if f.endswith((".png", ".jpg", ".jpeg"))]

    if max_images is not None:
        image_files = image_files[:max_images]

    num_images = len(image_files)
    features = torch.empty((num_images, 2048), device=device)

    for idx, image_file in enumerate(image_files):
        if verbose and (idx + 1) % 50 == 0:
            print(f"Processing: {idx + 1}/{num_images}")

        image_path = os.path.join(images_path, image_file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(image_tensor)
        features[idx] = feature

    if verbose:
        print(f"✓ Processed {num_images} images.")

    return features.cpu().numpy()


def extract_and_save_statistics(
    images_path: str,
    output_dir: str,
    dataset_name: str,
    max_images: int = None,
    device: str = None,
    verbose: bool = True,
) -> tuple[str, str]:
    """Extract Inception features from images and save mean vector and covariance matrix.

    This function processes images, extracts their features using Inception v3,
    computes the mean vector and covariance matrix, and saves them as .npy files.
    These statistics can later be used for efficient FID calculation.

    Args:
        images_path: Path to directory containing images
        output_dir: Directory where statistics will be saved
        dataset_name: Name for the dataset (used in filenames)
        max_images: Maximum number of images to process (None = all)
        device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        verbose: Whether to print progress messages

    Returns:
        Tuple of (mean_path, cov_path) - paths to saved statistics files
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print("=" * 80)
        print(f"EXTRACTING STATISTICS FOR: {dataset_name}")
        print("=" * 80)
        print(f"Images path: {images_path}")
        print(f"Output directory: {output_dir}")
        if max_images:
            print(f"Max images: {max_images}")
        print("-" * 80)

    os.makedirs(output_dir, exist_ok=True)

    model = _load_inception_model(device, verbose)
    preprocess = _get_inception_preprocessing()

    features = _extract_features_from_directory(
        images_path, model, preprocess, device, max_images, verbose
    )

    if verbose:
        print("Computing mean and covariance...")

    mean = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)

    mean_path = os.path.join(output_dir, f"{dataset_name}_mean.npy")
    cov_path = os.path.join(output_dir, f"{dataset_name}_cov.npy")

    np.save(mean_path, mean)
    np.save(cov_path, cov)

    if verbose:
        print(f"✓ Saved mean vector to: {mean_path}")
        print(f"✓ Saved covariance matrix to: {cov_path}")
        print(f"  Mean shape: {mean.shape}")
        print(f"  Covariance shape: {cov.shape}")
        print("=" * 80)

    return mean_path, cov_path


def calculate_fid_from_statistics(
    mean_path_1: str,
    cov_path_1: str,
    mean_path_2: str,
    cov_path_2: str,
    verbose: bool = True,
) -> float:
    """Calculate FID from pre-computed statistics.

    This is the efficient way to compute FID when you have pre-computed
    mean vectors and covariance matrices from both datasets.

    Args:
        mean_path_1: Path to .npy file containing mean vector of first dataset
        cov_path_1: Path to .npy file containing covariance matrix of first dataset
        mean_path_2: Path to .npy file containing mean vector of second dataset
        cov_path_2: Path to .npy file containing covariance matrix of second dataset
        verbose: Whether to print progress messages

    Returns:
        FID score (float)
    """
    if verbose:
        print("=" * 80)
        print("CALCULATING FID FROM PRE-COMPUTED STATISTICS")
        print("=" * 80)
        print(f"Dataset 1 mean: {mean_path_1}")
        print(f"Dataset 1 cov: {cov_path_1}")
        print(f"Dataset 2 mean: {mean_path_2}")
        print(f"Dataset 2 cov: {cov_path_2}")
        print("-" * 80)

    mu1 = np.load(mean_path_1)
    sigma1 = np.load(cov_path_1)
    mu2 = np.load(mean_path_2)
    sigma2 = np.load(cov_path_2)

    if verbose:
        print("✓ Statistics loaded successfully.")
        print("Computing FID...")

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        if verbose:
            print(
                "  Note: Covariance matrix product had imaginary components (numerical precision)"
            )
        covmean = covmean.real

    fid_score = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

    if verbose:
        print(f"✓ FID score: {fid_score:.4f}")
        print("=" * 80)

    return fid_score


def extract_and_save_coco_statistics(
    output_dir: str,
    num_images: int = 5000,
    classes: list = None,
    device: str = None,
    verbose: bool = True,
    log_to_wandb: bool = None,
) -> tuple[str, str]:
    """Extract and save statistics from COCO-2017 validation set.

    This function loads images from COCO-2017 validation set using fiftyone,
    extracts their features, and saves the statistics for later FID calculation.

    Args:
        output_dir: Directory where statistics will be saved
        num_images: Number of images to load from COCO-2017 validation set
        classes: List of classes to filter COCO images. If None, uses all images.
                 Example: ["person"] to only get images with people
        device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        verbose: Whether to print progress messages
        log_to_wandb: Whether to log progress to wandb. If None, auto-detects if wandb is active.

    Returns:
        Tuple of (mean_path, cov_path) - paths to saved statistics files
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if log_to_wandb is None:
        try:
            import wandb

            log_to_wandb = wandb.run is not None
        except ImportError:
            log_to_wandb = False

    if verbose:
        print("=" * 80)
        print("EXTRACTING COCO-2017 TEST SET STATISTICS")
        print("=" * 80)
        print(f"Number of images: {num_images}")
        if classes:
            print(f"Filtering for classes: {classes}")
        print("-" * 80)

    dataset_kwargs = {
        "split": "test",
        "max_samples": num_images,
    }

    if classes:
        dataset_kwargs["label_types"] = ["detections", "segmentations"]
        dataset_kwargs["classes"] = classes

    dataset = fo.zoo.load_zoo_dataset("coco-2017", **dataset_kwargs)

    if verbose:
        print(f"✓ Loaded {len(dataset)} images from COCO-2017")

    coco_image_paths = [sample.filepath for sample in dataset]

    if verbose:
        print(f"✓ Extracted {len(coco_image_paths)} image paths")
        print("-" * 80)

    if log_to_wandb:
        import wandb

        wandb.log({"coco/num_images_loaded": len(coco_image_paths)})

    model = _load_inception_model(device, verbose)
    preprocess = _get_inception_preprocessing()

    if verbose:
        print("Extracting features from COCO images...")

    num_images_actual = len(coco_image_paths)
    features = torch.empty((num_images_actual, 2048), device=device)

    for idx, image_path in enumerate(coco_image_paths):
        if verbose and (idx + 1) % 100 == 0:
            print(f"Processing: {idx + 1}/{num_images_actual}")

        if log_to_wandb and (idx + 1) % 100 == 0:
            import wandb

            wandb.log(
                {
                    "coco/images_processed": idx + 1,
                    "coco/progress_percent": ((idx + 1) / num_images_actual) * 100,
                }
            )

        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(image_tensor)
        features[idx] = feature

    if verbose:
        print(f"✓ Processed {num_images_actual} images.")

    if verbose:
        print("Computing mean and covariance...")

    if log_to_wandb:
        import wandb

        wandb.log({"coco/status": "computing_statistics"})

    features_np = features.cpu().numpy()
    mean = np.mean(features_np, axis=0)
    cov = np.cov(features_np, rowvar=False)

    os.makedirs(output_dir, exist_ok=True)

    dataset_name = f"coco2017_val_{num_images_actual}"
    if classes:
        classes_str = "_".join(classes[:3])
        dataset_name += f"_{classes_str}"

    mean_path = os.path.join(output_dir, f"{dataset_name}_mean.npy")
    cov_path = os.path.join(output_dir, f"{dataset_name}_cov.npy")

    np.save(mean_path, mean)
    np.save(cov_path, cov)

    if verbose:
        print(f"✓ Saved mean vector to: {mean_path}")
        print(f"✓ Saved covariance matrix to: {cov_path}")
        print(f"  Mean shape: {mean.shape}")
        print(f"  Covariance shape: {cov.shape}")
        print("=" * 80)

    if log_to_wandb:
        import wandb

        wandb.log(
            {
                "coco/status": "completed",
                "coco/mean_shape": list(mean.shape),
                "coco/cov_shape": list(cov.shape),
                "coco/dataset_name": dataset_name,
            }
        )

    return mean_path, cov_path


def calculate_fid(
    images_first_set_path: str,
    images_second_set_path: str,
    device: str = None,
    verbose: bool = True,
) -> float:
    """
    Calculate FID between two image sets.

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

    model = _load_inception_model(device, verbose)
    preprocess = _get_inception_preprocessing()

    if verbose:
        print("Processing first set...")
    features_1 = _extract_features_from_directory(
        images_first_set_path, model, preprocess, device, None, verbose
    )

    if verbose:
        print("Processing second set...")
    features_2 = _extract_features_from_directory(
        images_second_set_path, model, preprocess, device, None, verbose
    )

    if verbose:
        print("Computing statistics and FID...")

    mu1 = np.mean(features_1, axis=0)
    sigma1 = np.cov(features_1, rowvar=False)
    mu2 = np.mean(features_2, axis=0)
    sigma2 = np.cov(features_2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_score = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

    if verbose:
        print(f"✓ FID score: {fid_score:.4f}")
        print("=" * 80)

    return fid_score


def calculate_fid_with_coco(
    generated_images_path: str,
    num_coco_images: int = 150,
    device: str = None,
    verbose: bool = True,
    classes: list = None,
    coco_stats_dir: str = None,
) -> float:
    """
    Calculate FID between generated images and COCO-2017 validation set.

    This function can use pre-computed COCO statistics if available, or compute them on-the-fly.
    For better efficiency, pre-compute COCO statistics using extract_and_save_coco_statistics().

    Args:
        generated_images_path: Path to directory containing generated images
        num_coco_images: Number of images to load from COCO-2017 validation set
        device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        verbose: Whether to print progress messages
        classes: List of classes to filter COCO images. If None, uses all images.
                 Example: ["person"] to only get images with people
        coco_stats_dir: Directory containing pre-computed COCO statistics.
                       If None or stats not found, will compute on-the-fly.

    Returns:
        FID score (float)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    import tempfile

    if coco_stats_dir is not None:
        dataset_name = f"coco2017_val_{num_coco_images}"
        if classes:
            classes_str = "_".join(classes[:3])
            dataset_name += f"_{classes_str}"

        coco_mean_path = os.path.join(coco_stats_dir, f"{dataset_name}_mean.npy")
        coco_cov_path = os.path.join(coco_stats_dir, f"{dataset_name}_cov.npy")

        if os.path.exists(coco_mean_path) and os.path.exists(coco_cov_path):
            if verbose:
                print("=" * 80)
                print("USING PRE-COMPUTED COCO STATISTICS")
                print("=" * 80)
                print(f"COCO mean: {coco_mean_path}")
                print(f"COCO cov: {coco_cov_path}")
                print("-" * 80)

            temp_stats_dir = tempfile.mkdtemp(prefix="fid_stats_")
            try:
                gen_mean_path, gen_cov_path = extract_and_save_statistics(
                    images_path=generated_images_path,
                    output_dir=temp_stats_dir,
                    dataset_name="generated",
                    device=device,
                    verbose=verbose,
                )

                fid_score = calculate_fid_from_statistics(
                    mean_path_1=gen_mean_path,
                    cov_path_1=gen_cov_path,
                    mean_path_2=coco_mean_path,
                    cov_path_2=coco_cov_path,
                    verbose=verbose,
                )

                return fid_score

            finally:
                import shutil

                shutil.rmtree(temp_stats_dir, ignore_errors=True)
        elif verbose:
            print(f"Pre-computed COCO statistics not found at {coco_stats_dir}")
            print("Will compute COCO statistics on-the-fly...")

    if verbose:
        print("=" * 80)
        print("LOADING COCO-2017 VALIDATION SET")
        print("=" * 80)
        print(f"Loading {num_coco_images} images from COCO-2017")
        if classes:
            print(f"Filtering for classes: {classes}")
        print("-" * 80)

    dataset_kwargs = {
        "split": "validation",
        "max_samples": num_coco_images,
    }

    if classes:
        dataset_kwargs["label_types"] = ["detections", "segmentations"]
        dataset_kwargs["classes"] = classes

    dataset = fo.zoo.load_zoo_dataset("coco-2017", **dataset_kwargs)

    if verbose:
        print(f"✓ Loaded {len(dataset)} images from COCO-2017")

    coco_image_paths = [sample.filepath for sample in dataset]

    if verbose:
        print(f"✓ Extracted {len(coco_image_paths)} image paths")
        print("-" * 80)

    model = _load_inception_model(device, verbose)
    preprocess = _get_inception_preprocessing()

    if verbose:
        print("Extracting features from generated images...")
    gen_features = _extract_features_from_directory(
        generated_images_path, model, preprocess, device, None, verbose
    )

    if verbose:
        print("Extracting features from COCO images...")

    num_coco = len(coco_image_paths)
    coco_features = torch.empty((num_coco, 2048), device=device)

    for idx, image_path in enumerate(coco_image_paths):
        if verbose and (idx + 1) % 100 == 0:
            print(f"Processing: {idx + 1}/{num_coco}")

        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(image_tensor)
        coco_features[idx] = feature

    if verbose:
        print(f"✓ Processed {num_coco} COCO images.")

    if verbose:
        print("Computing statistics and FID...")

    mu1 = np.mean(gen_features, axis=0)
    sigma1 = np.cov(gen_features, rowvar=False)
    mu2 = np.mean(coco_features.cpu().numpy(), axis=0)
    sigma2 = np.cov(coco_features.cpu().numpy(), rowvar=False)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_score = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

    if verbose:
        print(f"✓ FID score: {fid_score:.4f}")
        print("=" * 80)

    return fid_score
