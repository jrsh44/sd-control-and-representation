#!/usr/bin/env python3
"""
Train a Sparse Autoencoder (SAE) on cached representations from Stable Diffusion layers.

This script loads datasets via RepresentationDataset, creates or loads an existing SAE model,
trains it on the training set with optional validation, and saves the updated model.

Validation is optional - if --test_dataset_path is not provided, training will proceed
without validation.

EXAMPLE USAGE:

    uv run scripts/sae/train.py \
        --train_dataset_path /mnt/evafs/groups/mi2lab/mjarosz/results/finetuned_sd_saeuron/cached_representations/unet_up_1_att_1 \
        --sae_path results/sae/unet_up_1_att_1_sae.pt \
        --expansion_factor 16 \
        --top_k 32 \
        --learning_rate 1e-4 \
        --num_epochs 5 \
        --batch_size 4096
"""

import argparse
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from overcomplete.sae import TopKSAE
from torch.utils.data import DataLoader

import wandb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from src.data.dataset import RepresentationDataset  # noqa: E402
from src.models.sae.training import criterion_laux, train_sae_val  # noqa: E402
from src.utils.wandb import get_system_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAE on representations from SD layer using RepresentationDataset."
    )
    # Dataset parameters
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        required=True,
        help="Path to training dataset directory (e.g., results/sd_1_5/unet_mid_att)",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        required=False,
        default=None,
        help="Path to test dataset directory (e.g., results/sd_1_5/unet_mid_att). Optional.",
    )
    # SAE parameters
    parser.add_argument(
        "--sae_path",
        type=str,
        required=True,
        help="Path to .pt file for SAE weights (load if exists, create and save if not)",
    )
    parser.add_argument(
        "--expansion_factor",
        type=int,
        default=16,
        help="SAE expansion factor",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=32,
        help="SAE top-k sparsity",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="SAE learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of SAE training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for SAE training",
    )
    parser.add_argument("--skip-wandb", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("LOAD DATASET AND TRAIN SAE ON REPRESENTATIONS")
    print("=" * 80)
    print(f"Train dataset path: {args.train_dataset_path}")
    print(f"Test dataset path: {args.test_dataset_path}")
    print(f"SAE path: {args.sae_path}")
    print(f"Device: {device}")
    print(
        f"SAE Config: epochs={args.num_epochs}, lr={args.learning_rate}, "
        f"exp_factor={args.expansion_factor}, top_k={args.top_k}, batch_size={args.batch_size}"
    )
    print("-" * 80)

    try:
        # Initialize wandb
        if not args.skip_wandb:
            wandb.login()
            wandb.init(
                project="sd-control-representation",
                entity="bartoszjezierski28-warsaw-university-of-technology",
                name=f"SAE_{Path(args.sae_path).stem}",
                config={
                    "sae_path": args.sae_path,
                    "val_dataset_path": args.test_dataset_path,
                    "expansion_factor": args.expansion_factor,
                    "top_k": args.top_k,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                },
                tags=["sae", "incremental", "validation"],
                notes="Trained incrementally on multiple train datasets, single validation set.",
            )

        # Load dataset
        training_dataset = RepresentationDataset(
            dataset_path=Path(args.train_dataset_path),
            return_metadata=False,
        )
        input_dim = training_dataset.feature_dim
        print(f"Loaded training dataset: {len(training_dataset)} samples, input_dim={input_dim}")

        # Load validation dataset if provided
        validation_dataset = None
        if args.test_dataset_path:
            validation_dataset = RepresentationDataset(
                dataset_path=Path(args.test_dataset_path),
                return_metadata=False,
            )
            print(f"Loaded validation dataset: {len(validation_dataset)} samples")
        else:
            print("No validation dataset provided - training without validation")

        # Prepare DataLoader
        is_cuda = device == "cuda"
        train_dataloader = DataLoader(
            training_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=is_cuda,
            num_workers=32 if is_cuda else 0,
            prefetch_factor=8 if is_cuda else None,
            persistent_workers=is_cuda,
        )

        # Create validation dataloader only if validation dataset exists
        val_dataloader = None
        if validation_dataset is not None:
            val_dataloader = DataLoader(
                validation_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=is_cuda,
                num_workers=32 if is_cuda else 0,
                prefetch_factor=8 if is_cuda else None,
                persistent_workers=is_cuda,
            )

        # Initialize or load SAE
        nb_concepts = input_dim * args.expansion_factor
        sae = TopKSAE(
            input_dim,
            nb_concepts=nb_concepts,
            top_k=args.top_k,
            device=device,
        )
        sae = sae.to(device)

        sae_path = Path(args.sae_path)
        if sae_path.exists():
            print(f"Loading existing SAE from: {sae_path}")
            sae.load_state_dict(torch.load(sae_path, map_location=device))
        else:
            print(f"Creating new SAE (file does not exist: {sae_path})")

        print("Compiling model with torch.compile() (first epoch will be slower)...")
        sae = torch.compile(sae)
        print("✓ Model compiled successfully")

        # Define optimizer
        optimizer = torch.optim.Adam(sae.parameters(), lr=args.learning_rate)

        # Train SAE
        print("\nStarting SAE training...")
        results_logs = train_sae_val(
            model=sae,
            train_dataloader=train_dataloader,
            criterion=criterion_laux,
            optimizer=optimizer,
            val_dataloader=val_dataloader,
            nb_epochs=args.num_epochs,
            device=device,
            use_amp=True,
            log_interval=10,
        )
        sae = sae.eval()
        print("\n✓ SAE training completed.")

        # Log to wandb
        if not args.skip_wandb:
            system_metrics_end = get_system_metrics(device)

            for epoch in range(args.num_epochs):
                log_dict = {
                    "epoch": epoch + 1,
                    "train/avg_loss": results_logs["train"]["avg_loss"][epoch],
                    "train/r2": results_logs["train"]["r2"][epoch],
                    "train/z_sparsity": results_logs["train"]["z_sparsity"][epoch],
                    "train/dead_features": results_logs["train"]["dead_features"][epoch],
                    "train/time_epoch": results_logs["train"]["time_epoch"][epoch],
                    "train_dataset": Path(args.train_dataset_path).name,
                }

                # Add validation metrics only if validation was performed
                if "val" in results_logs and results_logs["val"] is not None:
                    log_dict.update(
                        {
                            "val/avg_loss": results_logs["val"]["avg_loss"][epoch],
                            "val/r2": results_logs["val"]["r2"][epoch],
                            "val/z_sparsity": results_logs["val"]["z_sparsity"][epoch],
                            "val/dead_features": results_logs["val"]["dead_features"][epoch],
                            "val_dataset": Path(args.test_dataset_path).name,
                        }
                    )

                log_dict.update(system_metrics_end)
                wandb.log(log_dict)

        # Save the trained model
        sae_path.parent.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print("SAVING MODEL")
        print("=" * 80)
        print(f"Saving trained SAE model to: {sae_path}")
        torch.save(sae.state_dict(), sae_path)
        print("✓ Model saved successfully")

        print("\n" + "=" * 80)
        print("Done.")
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
