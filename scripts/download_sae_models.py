"""
Download SAE models from HuggingFace to local models directory.

This script downloads SAE models from the HuggingFace repository and organizes them
in the local file structure, making them available for all scripts that expect local paths.

Usage:
    # Download all available models
    uv run scripts/download_sae_models.py --all

    # Download specific experiments
    uv run scripts/download_sae_models.py --experiments experiment1 experiment2

    # Download to custom directory
    uv run scripts/download_sae_models.py --all --output-dir path/to/models

    # List available models without downloading
    uv run scripts/download_sae_models.py --list
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO_ID = "jrsh/sd-control-and-representation"
HF_REPO_TYPE = "model"

AVAILABLE_EXPERIMENTS = {"sd-v1-5": {"sae": ["exp32_topk16", "exp36_topk32", "exp36_topk64"]}}


def list_available_models():
    """List all available models in the HuggingFace repository."""
    print(f"\n📦 Available models in {HF_REPO_ID}:\n")

    for model_version, categories in AVAILABLE_EXPERIMENTS.items():
        print(f"  {model_version}/")
        for category, experiments in categories.items():
            print(f"    {category}/")
            for exp in experiments:
                print(f"      • {exp}")
                print("        - model.pt")
                print("        - config.json")
                print("        - merged_feature_sums.pt")
    print()


def download_experiment(
    experiment_name: str,
    model_version: str = "sd-v1-5",
    output_dir: Path = None,
    category: str = "sae",
) -> bool:
    """
    Download a single experiment from HuggingFace.

    Args:
        experiment_name: Name of the experiment (e.g., 'exp36_topk32')
        model_version: Model version (default: 'sd-v1-5')
        output_dir: Output directory (default: ./models)
        category: Category (default: 'sae')

    Returns:
        True if successful, False otherwise
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "models"

    # Create target directory structure
    target_dir = output_dir / model_version / category / experiment_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Files to download
    files_to_download = [
        ("model.pt", "SAE model weights"),
        ("config.json", "Model configuration"),
        ("merged_feature_sums.pt", "Feature statistics"),
    ]

    subfolder = f"{model_version}/{category}/{experiment_name}"

    print(f"\n📥 Downloading {experiment_name}...")
    print(f"   Target: {target_dir}")

    success = True
    for filename, description in files_to_download:
        try:
            print(f"   • {description} ({filename})...", end=" ", flush=True)

            # Download from HuggingFace (uses cache)
            cached_path = hf_hub_download(
                repo_id=HF_REPO_ID, subfolder=subfolder, filename=filename, repo_type=HF_REPO_TYPE
            )

            # Copy to target directory
            target_file = target_dir / filename

            # Read from cache and write to target
            import shutil

            shutil.copy2(cached_path, target_file)

            print("✓")

        except Exception as e:
            print(f"✗ ({e})")
            success = False

    if success:
        print(f"   ✅ {experiment_name} downloaded successfully!")

        # Display info from config
        config_file = target_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                print(
                    f"   📊 Config: top_k={config.get('top_k', 'N/A')}, "
                    f"nb_concepts={config.get('nb_concepts', 'N/A')}"
                )
            except Exception:
                pass

    return success


def download_all_models(output_dir: Path = None):
    """Download all available models."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "models"

    print(f"\n🚀 Downloading all SAE models to: {output_dir.absolute()}\n")

    total_success = 0
    total_attempted = 0

    for model_version, categories in AVAILABLE_EXPERIMENTS.items():
        for category, experiments in categories.items():
            for experiment in experiments:
                total_attempted += 1
                if download_experiment(experiment, model_version, output_dir, category):
                    total_success += 1

    print(f"\n{'=' * 60}")
    print(f"✅ Successfully downloaded: {total_success}/{total_attempted} models")
    print(f"📁 Location: {output_dir.absolute()}")
    print(f"{'=' * 60}\n")


def verify_local_models(models_dir: Path = None):
    """Verify which models are already downloaded locally."""
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "models"

    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return

    print(f"\n📂 Checking local models in: {models_dir.absolute()}\n")

    found_any = False
    for model_version, categories in AVAILABLE_EXPERIMENTS.items():
        version_dir = models_dir / model_version
        if not version_dir.exists():
            continue

        for category, experiments in categories.items():
            category_dir = version_dir / category
            if not category_dir.exists():
                continue

            for experiment in experiments:
                exp_dir = category_dir / experiment
                if not exp_dir.exists():
                    continue

                # Check for required files
                model_file = exp_dir / "model.pt"
                config_file = exp_dir / "config.json"
                sums_file = exp_dir / "merged_feature_sums.pt"

                if all(f.exists() for f in [model_file, config_file, sums_file]):
                    found_any = True
                    print(f"  ✅ {model_version}/{category}/{experiment}")

                    # Show file sizes
                    model_size_mb = model_file.stat().st_size / (1024 * 1024)
                    sums_size_mb = sums_file.stat().st_size / (1024 * 1024)
                    print(f"     Model: {model_size_mb:.1f} MB, Features: {sums_size_mb:.1f} MB")
                else:
                    print(f"  ⚠️  {model_version}/{category}/{experiment} (incomplete)")

    if not found_any:
        print("  No models found. Use --all to download.")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download SAE models from HuggingFace to local directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--all", action="store_true", help="Download all available models")

    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Download specific experiments (e.g., exp36_topk32 exp32_topk16)",
    )

    parser.add_argument(
        "--model-version", default="sd-v1-5", help="Model version (default: sd-v1-5)"
    )

    parser.add_argument("--output-dir", type=Path, help="Output directory (default: ./models)")

    parser.add_argument(
        "--list", action="store_true", help="List available models without downloading"
    )

    parser.add_argument(
        "--verify", action="store_true", help="Verify which models are already downloaded locally"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # List available models
    if args.list:
        list_available_models()
        return 0

    # Verify local models
    if args.verify:
        verify_local_models(args.output_dir)
        return 0

    # Download models
    if args.all:
        download_all_models(args.output_dir)
    elif args.experiments:
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "models"

        for exp in args.experiments:
            download_experiment(exp, args.model_version, output_dir)
    else:
        print("❌ Error: Please specify --all, --experiments, --list, or --verify")
        print("   Use --help for more information")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
