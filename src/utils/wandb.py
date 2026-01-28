import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import GPUtil
import psutil
import wandb


def get_system_metrics(device: str) -> Dict:
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    gpu_metrics = {}
    if device == "cuda":
        try:
            gpu = GPUtil.getGPUs()[0]
            gpu_metrics = {
                "gpu_memory_used": gpu.memoryUsed,
                "gpu_memory_total": gpu.memoryTotal,
                "gpu_load": gpu.load * 100,
            }
        except Exception:
            print("Warning: Could not retrieve GPU metrics.")

    return {"cpu_percent": cpu_percent, "memory_percent": memory.percent, **gpu_metrics}


def export_runs_to_csv(entity: str, project: str, output_file: str) -> None:
    """
    Export all runs from a wandb project to a CSV file.

    Args:
        entity: wandb entity
        project: wandb project name
        output_file: output CSV file path
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    runs_data = []
    for run in runs:
        runs_data.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "runtime": run.summary.get("_wandb", {}).get("runtime"),
                "user": str(run.user).split()[-1].rstrip(">") if run.user else None,
            }
        )

    if runs_data:
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=runs_data[0].keys())
            writer.writeheader()
            writer.writerows(runs_data)

        print(f"Exported {len(runs_data)} runs to {output_file}")
    else:
        print(f"No runs found in {entity}/{project}")


def download_images_from_run(
    entity: str,
    project: str,
    run_id: str,
    output_dir: str,
    log_every: int = 50,
) -> List[str]:
    """
    Download all logged images from a specific wandb run using parallel downloads.
    Images are saved with their original filenames from wandb media folder.

    Args:
        entity: wandb entity
        project: wandb project name
        run_id: specific run ID
        output_dir: directory to save images
        log_every: log progress every N downloaded files (default: 50)

    Returns:
        List of downloaded image paths
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    downloaded_images = []

    print("  Downloading images from media files...")

    def download_single_image(file):
        """Download a single image file."""
        try:
            image_filename = Path(file.name).name
            file.download(root=str(output_path), replace=True)
            local_path = output_path / image_filename
            return str(local_path), image_filename, None
        except Exception as e:
            return None, file.name, str(e)

    try:
        files = run.files()
        image_files = [
            file
            for file in files
            if file.name.startswith("media/images/")
            and file.name.endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_files:
            print("  No image files found in media/images/")
            return downloaded_images

        num_cpus = len(os.sched_getaffinity(0))
        num_workers = max(num_cpus - 1, 1)
        print(f"  Available CPUs: {num_cpus}, using {num_workers} workers")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(download_single_image, file): file for file in image_files
            }

            for idx, future in enumerate(as_completed(future_to_file), 1):
                local_path, filename, error = future.result()
                if error:
                    print(f"    ✗ Failed to download {filename}: {error}")
                else:
                    downloaded_images.append(local_path)
                    if idx % log_every == 0 or idx == len(image_files):
                        print(f"    ✓ Progress: {idx}/{len(image_files)} files downloaded")

    except Exception as e:
        print(f"  Warning: Could not retrieve image files: {e}")
        import traceback

        traceback.print_exc()

    print(f"  Total images downloaded: {len(downloaded_images)}")
    return downloaded_images
