#!/usr/bin/env python3
"""
Train a Sparse Autoencoder (SAE) on representations from a Stable Diffusion layer using RepresentationDataset.

This script loads a dataset via RepresentationDataset, creates or loads an existing SAE model from a specified .pt file,
trains it on the dataset, and saves the updated model back to the .pt file.

Designed for incremental training across multiple datasets by reusing the same SAE model path.
"""

import os
import wandb
import argparse
from pathlib import Path
import sys
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
from overcomplete.sae import TopKSAE, train_sae
from src.models.sae.training import criterion_laux
from src.data.dataset import RepresentationDataset
from src.utils.wandb import get_system_metrics
from models.sae.train_sae_validation import train_sae_val

# Add project root to path to allow imports to work from any location
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env from project root
load_dotenv(dotenv_path=project_root / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAE on representations from Stable Diffusion layer using RepresentationDataset."
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
        required=True,
        help="Path to training dataset directory (e.g., results/sd_1_5/unet_mid_att)",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        default=True,
        help="Flatten representations to 1D vectors (default: True)",
    )
    parser.add_argument(
        "--return_metadata",
        action="store_true",
        default=True,
        help="Return metadata along with representations (default: True)",
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
    parser.add_argument(
        "--skip-wandb",
        action="store_true"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("LOAD DATASET AND TRAIN SAE ON REPRESENTATIONS")
    print("=" * 80)
    print(f"Train dataset path: {args.train_dataset_path}")
    print(f"Test dataset path: {args.test_dataset_path}")
    print(f"Flatten: {args.flatten}")
    print(f"Return metadata: {args.return_metadata}")
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
                name=f"SAE_{Path(args.sae_path).stem}",  # nazwa eksperymentu
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
        trining_dataset = RepresentationDataset(
            dataset_path=Path(args.train_dataset_path),
            flatten=args.flatten,
            return_metadata=args.return_metadata,
        )
        validation_dataset = RepresentationDataset(
            dataset_path=Path(args.test_dataset_path),
            flatten=args.flatten,
            return_metadata=args.return_metadata,
        )
        input_dim = trining_dataset.feature_dim
        print(f"Loaded dataset: {len(trining_dataset)} samples, input_dim={input_dim}")

        # Prepare DataLoader
        train_dataloader = DataLoader(trining_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

        # Initialize or load SAE
        nb_concepts = input_dim * args.expansion_factor
        sae = TopKSAE(input_dim, nb_concepts=nb_concepts, top_k=args.top_k, device=device)
        sae = sae.to(device)

        sae_path = Path(args.sae_path)
        if sae_path.exists():
            print(f"Loading existing SAE from: {sae_path}")
            sae.load_state_dict(torch.load(sae_path, map_location=device))
        else:
            print(f"Creating new SAE (file does not exist: {sae_path})")

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
        )
        # train_sae(sae, dataloader, criterion_laux, optimizer, nb_epochs=args.num_epochs, device=device)
        sae = sae.eval()
        print("\n✓ SAE training completed.")

        # Log to wandb
        if not args.skip_wandb:
            system_metrics_end = get_system_metrics(device)

            for epoch in range(args.num_epochs):
                log_dict = {
                    "epoch": epoch + 1,
                    "train/avg_loss": results_logs['train']['avg_loss'][epoch],
                    "train/r2": results_logs['train']['r2'][epoch],
                    "train/z_sparsity": results_logs['train']['z_sparsity'][epoch],
                    "train/dead_features": results_logs['train']['dead_features'][epoch],
                    "train/time_epoch": results_logs['train']['time_epoch'][epoch],

                    "val/avg_loss": results_logs['val']['avg_loss'][epoch],
                    "val/r2": results_logs['val']['r2'][epoch],
                    "val/z_sparsity": results_logs['val']['z_sparsity'][epoch],
                    "val/dead_features": results_logs['val']['dead_features'][epoch],

                    "train_dataset": Path(args.train_dataset_path).name,
                    "val_dataset": Path(args.test_dataset_path).name,
                }
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