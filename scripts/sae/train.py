#!/usr/bin/env python3
"""
Train a Sparse Autoencoder (SAE) on cached representations from Stable Diffusion layers.

This script loads datasets via RepresentationDataset, creates or loads an existing SAE model,
trains it on the training set with optional validation, and saves the updated model.

Validation is optional - if --test_dataset_path is not provided, training will proceed
without validation.

EXAMPLE USAGE:

    uv run scripts/sae/train.py         \
    --train_dataset_path /mnt/evafs/groups/mi2lab/bjezierski/results/finetuned_sd_saeuron/cached_representations/unet_up_1_att_1 \
    --sae_path ../results/sae/unet_up_1_att_1_sae.pt         \
    --expansion_factor 16         \
    --top_k 32         \
    --learning_rate 4e-4         \
    --num_epochs 5         \
    --batch_size 4096         \
    --log_interval 10
"""  # noqa: E501

import argparse
import platform
import sys
from pathlib import Path

import torch
import torch.optim as optim
from dotenv import load_dotenv
from torch.utils.data import DataLoader

import wandb

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from overcomplete.sae import TopKSAE  # noqa: E402
from torch.amp.grad_scaler import GradScaler  # noqa: E402

from src.data.dataset import RepresentationDataset  # noqa: E402
from src.models.sae.training import (
    criterion_laux,
    train_sae_val,
)  # noqa: E402


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
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log progress every N batches (default: 10)",
    )
    parser.add_argument("--skip-wandb", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device == "cuda"

    # --------------------------------------------------------------------------
    # 1. Load Dataset
    # --------------------------------------------------------------------------
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    print(f"Train dataset path: {args.train_dataset_path}")

    train_path = Path(args.train_dataset_path)
    train_cache_dir = train_path.parent
    train_layer_name = train_path.name

    print("\n⚡ Using memmap cache for fast loading...")
    train_dataset = RepresentationDataset(
        cache_dir=train_cache_dir,
        layer_name=train_layer_name,
        return_metadata=False,
    )
    input_dim = train_dataset._full_data.shape[1]
    print(f"Dataset loaded. Input dim: {input_dim}")

    # --------------------------------------------------------------------------
    # 2. Initialize WandB
    # --------------------------------------------------------------------------
    if not args.skip_wandb:
        wandb.login()

        # Create a run name
        run_name = f"SAE_{train_layer_name}_exp{args.expansion_factor}_k{args.top_k}"

        gpu_name = torch.cuda.get_device_name(0) if is_cuda else "Unknown"

        # Structured Configuration
        config = {
            "dataset": {
                "layer_name": train_layer_name,
                "train_path": str(train_path),
                "val_path": str(args.test_dataset_path) if args.test_dataset_path else "None",
                "input_dim": input_dim,
                "total_samples": len(train_dataset),
            },
            "model": {
                "architecture": "TopKSAE",
                "expansion_factor": args.expansion_factor,
                "top_k": args.top_k,
                "num_concepts": input_dim * args.expansion_factor,
                "input_dim": input_dim,
            },
            "training": {
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.num_epochs,
                "optimizer": "Adam",
                "grad_clip": 1.0,
                "scheduler": "None",
            },
            "hardware": {
                "device": str(device),
                "gpu_name": gpu_name,
                "platform": platform.platform(),
                "python_version": sys.version.split()[0],
                "node": platform.node(),
            },
        }

        wandb.init(
            project="sd-control-representation",
            entity="bartoszjezierski28-warsaw-university-of-technology",
            name=run_name,
            config=config,
            group=train_layer_name,
            job_type="train",
            tags=["sae", "train", train_layer_name, f"exp{args.expansion_factor}"],
            notes="Trained incrementally on cached representations.",
        )
        print(f"🚀 WandB Initialized: {run_name}")

    # --------------------------------------------------------------------------
    # 3. Setup Model & DataLoaders
    # --------------------------------------------------------------------------
    if is_cuda:
        import os

        num_cpus = len(os.sched_getaffinity(0))
        num_workers = min(max(num_cpus - 1, 1), 16)
        print(f"Available CPUs: {num_cpus}, using {num_workers} workers")
    else:
        num_workers = 0

    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    val_dl = None
    if args.test_dataset_path:
        val_path = Path(args.test_dataset_path)
        val_dataset = RepresentationDataset(
            cache_dir=val_path.parent, layer_name=val_path.name, return_metadata=False
        )
        val_dl = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"Loaded validation dataset: {len(val_dataset)} samples")

    nb_concepts = input_dim * args.expansion_factor
    sae = TopKSAE(input_dim, nb_concepts=nb_concepts, top_k=args.top_k, device=device)
    sae = sae.to(device)

    # Load existing if available
    sae_path = Path(args.sae_path)
    if sae_path.exists():
        print(f"Loading weights from {sae_path}")
        sae.load_state_dict(torch.load(sae_path, map_location=device))
    else:
        print(f"Creating new SAE: {sae_path}")

    # Compile if supported
    if is_cuda:
        gpu_capability = torch.cuda.get_device_capability(0)
        if gpu_capability[0] >= 7:
            print("Compiling model with torch.compile()...")
            try:
                sae = torch.compile(sae)
                print("✓ Model compiled")
            except Exception as e:
                print(f"⚠️ Compilation failed: {e}")

    optimizer = optim.Adam(sae.parameters(), lr=args.learning_rate)
    scaler = GradScaler("cuda") if is_cuda else None

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    train_sae_val(
        model=sae,
        train_dataloader=train_dl,
        criterion=criterion_laux,
        optimizer=optimizer,
        val_dataloader=val_dl,
        scheduler=None,
        nb_epochs=args.num_epochs,
        clip_grad=1.0,
        monitoring=1,
        device=device,
        use_amp=scaler is not None,
        log_interval=args.log_interval,
        wandb_enabled=not args.skip_wandb,
    )

    # Save final model
    sae_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sae.state_dict(), sae_path)
    print(f"Final model saved: {sae_path}")

    if not args.skip_wandb:
        wandb.finish()

    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
