"""
Script to calculate CLIP scores between prompts and images.

This script reads prompts from prompts_analysis.txt, fills them with concept names,
and calculates CLIP scores between the filled prompts and corresponding images.

Folder structure: {source_path}/{concept}/{intervention_folder}/prompt_{4_digits}.png
Prompts file format: prompt_nr;prompt text with {} placeholder

Example usage:
uv run scripts/image_evaluation/calculate_clip_prompt_image_scores.py \
  --source_folder /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/images \
  --prompts_file /mnt/evafs/groups/mi2lab/bjezierski/sd-control-and-representation/data/nudity/prompts_analysis.txt \
  --output_csv /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/clip_prompt_image_scores.csv
"""

import argparse
import csv
import re
import sys
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


def load_prompts(prompts_file: Path) -> Dict[int, str]:
    """
    Load prompts from file.

    Args:
        prompts_file: Path to prompts file (format: prompt_nr;text)

    Returns:
        Dictionary mapping prompt number to prompt text
    """
    prompts = {}
    with open(prompts_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";", 1)
            if len(parts) == 2:
                prompt_nr = int(parts[0])
                prompt_text = parts[1]
                prompts[prompt_nr] = prompt_text
    return prompts


def format_concept_name(concept_folder_name: str) -> str:
    """
    Format concept folder name for prompt filling.

    Args:
        concept_folder_name: Folder name like 'exposed_anus' or 'female_breast'

    Returns:
        Formatted concept name like 'exposed anus' or 'female breast'
    """
    # Replace underscores with spaces
    return concept_folder_name.replace("_", " ")


def parse_image_path(image_path: Path, source_folder: Path) -> Dict[str, any]:
    """
    Parse image path to extract intervention parameters and prompt number.

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
    else:
        # Extract intervention parameters
        match = re.match(r"fn_?(\d+)_?if_?([\d.]+)", intervention_folder)
        if not match:
            raise ValueError(f"Invalid intervention folder format: {intervention_folder}")

        num_neurons = int(match.group(1))
        intervention_strength = float(match.group(2))

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
        "intervention_folder": intervention_folder,
    }


def get_all_images_from_folder(folder_path: Path) -> List[Path]:
    """Get all .png files from a folder."""
    return list(folder_path.glob("*.png"))


def evaluate_prompt_clip_scores(
    source_folder: str,
    prompts_file: str,
    output_csv: str,
) -> None:
    """
    Evaluate CLIP scores between prompts and images.

    Args:
        source_folder: Root folder containing intervention images
        prompts_file: Path to prompts file
        output_csv: Path to save CSV results
    """
    source_path = Path(source_folder)
    prompts_path = Path(prompts_file)

    if not source_path.exists():
        raise ValueError(f"Source folder does not exist: {source_folder}")

    if not prompts_path.exists():
        raise ValueError(f"Prompts file does not exist: {prompts_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load prompts
    print(f"Loading prompts from: {prompts_file}")
    prompts = load_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts")

    # Initialize CLIPScore metric
    print("Loading CLIP model from torchmetrics...")
    metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    print("✓ CLIP model loaded successfully")

    # Collect all images
    print(f"\nScanning for images in: {source_folder}")
    all_images = []

    for concept_folder in source_path.iterdir():
        if not concept_folder.is_dir():
            continue

        for intervention_folder in concept_folder.iterdir():
            if not intervention_folder.is_dir():
                continue

            images = get_all_images_from_folder(intervention_folder)
            all_images.extend(images)

    print(f"Found {len(all_images)} images to process")

    # Prepare CSV file and write header
    print(f"Writing results to: {output_csv}")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "concept",
        "num_neurons",
        "intervention_strength",
        "prompt_number",
        "filled_prompt",
        "clip_score",
        "image_path",
    ]

    # Open CSV file and write header
    csvfile = open(output_csv, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()

    # Image transformation for CLIPScore
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).byte()),  # Convert to 0-255 range
        ]
    )

    # Process images sequentially
    num_processed = 0
    total_score = 0.0
    num_skipped = 0

    try:
        print("\nProcessing images...")
        for i, image_path in enumerate(all_images, 1):
            try:
                # Parse image path
                params = parse_image_path(image_path, source_path)
                prompt_number = params["prompt_number"]
                concept = params["concept"]

                # Get prompt template
                if prompt_number not in prompts:
                    print(f"Warning: No prompt for number {prompt_number}, skipping {image_path}")
                    num_skipped += 1
                    continue

                prompt_template = prompts[prompt_number]

                # Format concept name and fill prompt
                concept_name = format_concept_name(concept)
                filled_prompt = prompt_template.replace("{}", concept_name)

                # Load and convert image to tensor
                img = Image.open(image_path).convert("RGB")
                img_tensor = transform(img)

                # Calculate CLIP score between prompt and image
                score = metric(img_tensor, filled_prompt).item()

                # Create result
                result = {
                    "concept": concept,
                    "num_neurons": params["num_neurons"],
                    "intervention_strength": params["intervention_strength"],
                    "prompt_number": prompt_number,
                    "filled_prompt": filled_prompt,
                    "clip_score": score,
                    "image_path": str(image_path.relative_to(source_path)),
                }

                # Write result immediately to CSV
                writer.writerow(result)
                csvfile.flush()

                num_processed += 1
                total_score += score

                # Progress update
                if i % 10 == 0 or i == len(all_images):
                    print(f"Processed {i}/{len(all_images)} images...")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

    finally:
        # Always close the file
        csvfile.close()

    # Print summary
    print(f"\nSuccessfully saved {num_processed} results")
    print(f"Skipped {num_skipped} images (no matching prompt)")
    if num_processed > 0:
        print(f"Average CLIP score: {total_score / num_processed:.4f}")
    else:
        print("Warning: No results were processed successfully")


def main():
    parser = argparse.ArgumentParser(description="Calculate CLIP scores between prompts and images")
    parser.add_argument(
        "--source_folder",
        type=str,
        required=True,
        help="Root folder containing intervention images",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Path to prompts file (format: prompt_nr;text)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save CSV results",
    )

    args = parser.parse_args()

    evaluate_prompt_clip_scores(
        source_folder=args.source_folder,
        prompts_file=args.prompts_file,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
