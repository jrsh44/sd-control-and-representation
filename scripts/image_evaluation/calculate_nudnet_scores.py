"""
Script to evaluate nudity scores for intervention images.

This script processes images from a folder structure with intervention parameters
and calculates nudity scores using NudeNet.

Folder structure: {source_path}/{concept}/fn_{two_digits}if_{float}/prompt_{4_digits}.png

Usage:
    uv run scripts/image_evaluation/calculate_nudnet_scores.py \
        --source_folder path/to/images \
        --output_csv path/to/output.csv
"""

import argparse
import csv
import os
import re
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.nudenet import NudeNetWrapper  # noqa: E402


def get_all_images_from_folder(folder_path: Path) -> List[Path]:
    """
    Get all .png files from a folder.

    Args:
        folder_path: Path to the folder

    Returns:
        List of paths to .png files
    """
    return list(folder_path.glob("*.png"))


def parse_image_path(image_path: Path, source_folder: Path) -> Dict[str, any]:
    """
    Parse image path to extract intervention parameters.

    Expected structure:
    - {source_path}/{concept}/fn_{two_digits}if_{float}/prompt_{4_digits}.png
    - {source_path}/{concept}/no_intervention/prompt_{4_digits}.png (no intervention case)

    Args:
        image_path: Path to the image
        source_folder: Root source folder path

    Returns:
        Dictionary with parsed parameters: concept, num_neurons, intervention_strength, prompt_number
        For no_intervention folders, num_neurons and intervention_strength are None
    """
    # Get relative path from source folder
    relative_path = image_path.relative_to(source_folder)

    # Parse parts
    parts = relative_path.parts

    if len(parts) < 3:
        raise ValueError(f"Invalid path structure: {relative_path}")

    # Extract concept (first folder)
    concept = parts[0]

    # Extract intervention parameters from second folder
    intervention_folder = parts[1]

    # Check if this is a no_intervention folder
    if intervention_folder == "no_intervention":
        num_neurons = None
        intervention_strength = None
    else:
        # Extract intervention parameters: fn_{digits}_if_{float} or fn{digits}_if{float}
        match = re.match(r"fn_?(\d+)_?if_?([\d.]+)", intervention_folder)
        if not match:
            raise ValueError(f"Invalid intervention folder format: {intervention_folder}")

        num_neurons = int(match.group(1))
        intervention_strength = float(match.group(2))

    # Extract prompt number from filename: prompt_{4_digits}.png
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
    }


def process_single_image(
    image_path: Path,
    source_path: Path,
    model_name: str,
    model_path: Optional[str],
) -> Optional[Dict]:
    """
    Worker function to process a single image.

    Args:
        image_path: Path to the image
        source_path: Root source folder path
        model_name: NudeNet model name
        model_path: Path to custom model

    Returns:
        Dictionary with results or None if processing failed
    """
    try:
        # Initialize NudeNet wrapper in worker process
        nudenet = NudeNetWrapper(model_name=model_name, model_path=model_path)

        # Parse path to extract parameters
        params = parse_image_path(image_path, source_path)

        # Calculate nudity score filtered by concept
        concept = params["concept"]
        score = nudenet.score_image_by_concept(str(image_path), concept)

        # Return results
        return {
            "concept": params["concept"],
            "num_neurons": params["num_neurons"],
            "intervention_strength": params["intervention_strength"],
            "prompt_number": params["prompt_number"],
            "nudity_score": score,
            "image_path": str(image_path.relative_to(source_path)),
        }

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def evaluate_nudity_scores(
    source_folder: str,
    output_csv: str,
    model_name: str = "320n",
    model_path: str = None,
) -> None:
    """
    Evaluate nudity scores for all images in the source folder structure.

    Args:
        source_folder: Root folder containing the intervention images
        output_csv: Path to save the CSV results
        model_name: NudeNet model name ('320n' or '640m')
        model_path: Path to custom model (required for '640m')
    """
    source_path = Path(source_folder)

    if not source_path.exists():
        raise ValueError(f"Source folder does not exist: {source_folder}")

    # Set number of workers to available CPUs - 1
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        available_cpus = int(slurm_cpus)
    else:
        available_cpus = cpu_count() or 4

    num_workers = max(1, available_cpus - 1)

    print(f"Using {num_workers} parallel workers")

    # Collect all images
    print(f"Scanning for images in: {source_folder}")
    all_images = []

    # Traverse all subdirectories
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
        "nudity_score",
        "image_path",
    ]

    # Open CSV file and write header
    csvfile = open(output_csv, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()  # Ensure header is written immediately

    # Process images in parallel and write results incrementally
    worker_func = partial(
        process_single_image,
        source_path=source_path,
        model_name=model_name,
        model_path=model_path,
    )

    num_processed = 0
    total_score = 0.0

    try:
        with Pool(processes=num_workers) as pool:
            # Process images with progress tracking and incremental writing
            for i, result in enumerate(pool.imap(worker_func, all_images), 1):
                if result is not None:
                    # Write result immediately to CSV
                    writer.writerow(result)
                    csvfile.flush()  # Ensure it's written to disk

                    num_processed += 1
                    total_score += result["nudity_score"]

                if i % 10 == 0:
                    print(f"Processed {i}/{len(all_images)} images...")
    finally:
        # Always close the file
        csvfile.close()

    # Print summary
    print(f"\nSuccessfully saved {num_processed} results")
    if num_processed > 0:
        print(f"Average nudity score: {total_score / num_processed:.4f}")
    else:
        print("Warning: No results were processed successfully")


def main():
    parser = argparse.ArgumentParser(description="Evaluate nudity scores for intervention images")
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
    parser.add_argument(
        "--model_name",
        type=str,
        default="320n",
        choices=["320n", "640m"],
        help="NudeNet model name (default: 320n)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to custom model (required for 640m)",
    )

    args = parser.parse_args()

    evaluate_nudity_scores(
        source_folder=args.source_folder,
        output_csv=args.output_csv,
        model_name=args.model_name,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
