"""
Script to calculate CLIP cosine similarities between no_intervention and intervention images.

This script processes images from a folder structure with intervention parameters
and calculates CLIP embedding cosine similarities using the CLIP model from Stable Diffusion.

Folder structure: {source_path}/{concept}/fn_{digits}if_{float}/prompt_{4_digits}.png
                  {source_path}/{concept}/no_intervention/prompt_{4_digits}.png

Example usage:
uv run scripts/image_evaluation/calculate_clip_image_image_scores.py \
  --source_folder /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/images \
  --output_csv /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/clip_image_image_scores.csv
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def parse_image_path(image_path: Path, source_folder: Path) -> Dict[str, any]:
    """
    Parse image path to extract intervention parameters.

    Args:
        image_path: Path to the image
        source_folder: Root source folder path

    Returns:
        Dictionary with parsed parameters
    """
    relative_path = image_path.relative_to(source_folder)
    parts = relative_path.parts

    if len(parts) < 3:
        raise ValueError(f"Invalid path structure: {relative_path}")

    concept = parts[0]
    intervention_folder = parts[1]

    # Check if this is a no_intervention folder
    if intervention_folder == "no_intervention":
        num_neurons = None
        intervention_strength = None
        is_baseline = True
    else:
        # Extract intervention parameters
        match = re.match(r"fn_?(\d+)_?if_?([\d.]+)", intervention_folder)
        if not match:
            raise ValueError(f"Invalid intervention folder format: {intervention_folder}")

        num_neurons = int(match.group(1))
        intervention_strength = float(match.group(2))
        is_baseline = False

    # Extract prompt number from filename
    filename = parts[2]
    match = re.match(r"prompt_(\d{4})\.png", filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")

    prompt_number = int(match.group(1))

    return {
        "concept": concept,
        "num_neurons": num_neurons,
        "intervention_strength": intervention_strength,
        "prompt_number": prompt_number,
        "is_baseline": is_baseline,
        "intervention_folder": intervention_folder,
    }


def get_all_images_from_folder(folder_path: Path) -> List[Path]:
    """Get all .png files from a folder."""
    return list(folder_path.glob("*.png"))


def evaluate_clip_similarities(
    source_folder: str,
    output_csv: str,
) -> None:
    """
    Evaluate CLIP scores between no_intervention and intervention images.

    Args:
        source_folder: Root folder containing intervention images
        output_csv: Path to save CSV results
    """
    source_path = Path(source_folder)

    if not source_path.exists():
        raise ValueError(f"Source folder does not exist: {source_folder}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize CLIPScore metric
    print("Loading CLIP model from torchmetrics...")
    metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    print("✓ CLIP model loaded successfully")

    # Collect all images organized by concept and prompt
    print(f"Scanning for images in: {source_folder}")
    images_by_concept = defaultdict(lambda: defaultdict(dict))

    for concept_folder in source_path.iterdir():
        if not concept_folder.is_dir():
            continue

        concept = concept_folder.name

        for intervention_folder in concept_folder.iterdir():
            if not intervention_folder.is_dir():
                continue

            images = get_all_images_from_folder(intervention_folder)

            for image_path in images:
                try:
                    params = parse_image_path(image_path, source_path)
                    prompt_num = params["prompt_number"]

                    if params["is_baseline"]:
                        images_by_concept[concept][prompt_num]["baseline"] = image_path
                    else:
                        if "interventions" not in images_by_concept[concept][prompt_num]:
                            images_by_concept[concept][prompt_num]["interventions"] = []

                        images_by_concept[concept][prompt_num]["interventions"].append(
                            (image_path, params)
                        )
                except Exception as e:
                    print(f"Error parsing {image_path}: {e}")
                    continue

    # Create comparison pairs
    print("\nCreating comparison pairs...")
    comparisons = []

    for concept, prompts_dict in images_by_concept.items():
        for prompt_num, data in prompts_dict.items():
            baseline_path = data.get("baseline")
            interventions = data.get("interventions", [])

            if baseline_path is None:
                print(f"Warning: No baseline image for {concept}/prompt_{prompt_num:04d}")
                continue

            for intervention_path, params in interventions:
                comparison = (baseline_path, intervention_path, params)
                comparisons.append(comparison)

    print(f"Found {len(comparisons)} comparisons to process")

    if len(comparisons) == 0:
        print("No comparisons to process!")
        return

    # Prepare CSV file and write header
    print(f"Writing results to: {output_csv}")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "concept",
        "num_neurons",
        "intervention_strength",
        "prompt_number",
        "clip_score",
        "baseline_path",
        "intervention_path",
    ]

    # Open CSV file and write header
    csvfile = open(output_csv, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()

    # Image transformation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).byte()),  # Convert to 0-255 range
        ]
    )

    # Process comparisons sequentially
    num_processed = 0
    total_score = 0.0

    try:
        print("\nProcessing comparisons...")
        for i, (baseline_path, intervention_path, params) in enumerate(comparisons, 1):
            try:
                # Load images as tensors
                baseline_img = Image.open(baseline_path).convert("RGB")
                intervention_img = Image.open(intervention_path).convert("RGB")

                # Convert PIL images to tensors (C, H, W) format
                baseline_tensor = transform(baseline_img)
                intervention_tensor = transform(intervention_img)

                # Calculate CLIP score between images
                score = metric(baseline_tensor, intervention_tensor).item()

                # Create result
                result = {
                    "concept": params["concept"],
                    "num_neurons": params["num_neurons"],
                    "intervention_strength": params["intervention_strength"],
                    "prompt_number": params["prompt_number"],
                    "clip_score": score,
                    "baseline_path": str(baseline_path.relative_to(source_path)),
                    "intervention_path": str(intervention_path.relative_to(source_path)),
                }

                # Write result immediately to CSV
                writer.writerow(result)
                csvfile.flush()

                num_processed += 1
                total_score += score

                # Progress update
                if i % 10 == 0 or i == len(comparisons):
                    print(f"Processed {i}/{len(comparisons)} comparisons...")

            except Exception as e:
                print(f"Error processing {baseline_path} vs {intervention_path}: {e}")
                continue

    finally:
        # Always close the file
        csvfile.close()

    # Print summary
    print(f"\nSuccessfully saved {num_processed} results")
    if num_processed > 0:
        print(f"Average CLIP score: {total_score / num_processed:.4f}")
    else:
        print("Warning: No results were processed successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate CLIP scores between baseline and intervention images"
    )
    parser.add_argument(
        "--source_folder",
        type=str,
        required=True,
        help="Root folder containing intervention images",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save CSV results",
    )

    args = parser.parse_args()

    evaluate_clip_similarities(
        source_folder=args.source_folder,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
