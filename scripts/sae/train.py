#!/usr/bin/env python3
"""
Train a Sparse Autoencoder (SAE) on cached representations from Stable Diffusion layers.

This script loads datasets via RepresentationDataset, creates or loads an existing SAE model,
trains it on the training set with optional validation, and saves the updated model.

Supports:
- Multiple dataset paths (combined using ConcatDataset)
- Internal train/validation split via --validation_percent
- Separate validation dataset via --test_dataset_path

EXAMPLE USAGE:

    # Multiple datasets with validation split
    uv run scripts/sae/train.py \
        --dataset_paths /path/to/dataset1/layer /path/to/dataset2/layer \
        --sae_path results/sae/model.pt \
        --config_path results/sae/config.json \
        --datasets_name cc3m-wds_nudity \
        --expansion_factor 16 \
        --top_k 32 \
        --learning_rate 1e-4 \
        --num_epochs 5 \
        --batch_size 4096 \
        --validation_percent 10 \
        --validation_seed 42 \
        --skip-wandb

    # Single dataset with separate validation set (old style)
    uv run scripts/sae/train.py \
        --dataset_paths /path/to/train/layer \
        --test_dataset_path /path/to/val/layer \
        --sae_path results/sae/model.pt \
        --config_path results/sae/config.json \
        --datasets_name single_dataset \
        --expansion_factor 16 \
        --top_k 32 \
        --learning_rate 1e-4 \
        --num_epochs 5 \
        --batch_size 4096 \
        --skip-wandb
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from dotenv import load_dotenv
from overcomplete.sae import TopKSAE
from torch.utils.data import ConcatDataset, DataLoader, Sampler, Subset

import wandb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from src.data.dataset import RepresentationDataset  # noqa: E402
from src.models.sae.training.config import SchedulerConfig, TrainingConfig  # noqa: E402
from src.models.sae.training.trainer import SAETrainer  # noqa: E402
from src.models.sae.training.utils import create_warmup_cosine_scheduler  # noqa: E402
from src.utils.wandb import get_system_metrics  # noqa: E402


def get_optimal_dataloader_params(is_cuda: bool) -> dict:
    """
    Calculate optimal DataLoader parameters.

    For memmap datasets, the bottleneck is typically I/O (especially on network FS).
    More workers help with prefetching, but too many can cause memory pressure
    from Python interpreter overhead and page cache contention.

    Args:
        is_cuda: Whether using CUDA (GPU)

    Returns:
        dict with num_workers, prefetch_factor, pin_memory, persistent_workers
    """
    if not is_cuda:
        return {
            "num_workers": 0,
            "prefetch_factor": None,
            "pin_memory": False,
            "persistent_workers": False,
        }

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        available_cpus = int(slurm_cpus)
    else:
        available_cpus = os.cpu_count() or 4

    # Can override with DATALOADER_NUM_WORKERS environment variable
    num_workers_override = os.environ.get("DATALOADER_NUM_WORKERS")
    if num_workers_override:
        num_workers = int(num_workers_override)
    else:
        # Use most available CPUs, leave 2 for main process
        num_workers = max(2, available_cpus - 2)

    return {
        "num_workers": num_workers,
        "prefetch_factor": 4,
        "pin_memory": True,
        "persistent_workers": True,
    }


def get_checkpoint_path(sae_path: Path) -> Path:
    """Get the checkpoint file path based on sae_path."""
    return sae_path.parent / "checkpoint.pt"


def load_checkpoint(checkpoint_path: Path, device: str):
    """
    Load checkpoint if it exists and is valid.

    Returns:
        tuple: (checkpoint_dict, start_epoch, existing_logs) or (None, 0, None) if no checkpoint
    """
    if not checkpoint_path.exists():
        return None, 0, None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint.get("epoch", 0) + 1  # Resume from next epoch
        existing_logs = checkpoint.get("logs", None)
        print(f"  Found checkpoint at epoch {checkpoint['epoch'] + 1}")
        return checkpoint, start_epoch, existing_logs
    except Exception as e:
        print(f"  Warning: Failed to load checkpoint: {e}")
        return None, 0, None


def save_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    epoch: int,
    train_logs: dict,
    val_logs: dict,
    config: dict,
):
    """Save a training checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "logs": {"train": train_logs, "val": val_logs},
        "config": config,
    }

    # Save to temporary file first, then rename (atomic operation)
    temp_path = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.rename(checkpoint_path)
    print(f"   💾 Checkpoint saved (epoch {epoch + 1})")


def check_config_match(saved_config: dict, current_config: dict) -> bool:
    """Check if saved config matches current config (for resume validation)."""
    # Keys that must match for valid resume
    critical_keys = [
        "expansion_factor",
        "top_k",
        "learning_rate",
        "batch_size",
        "datasets_name",
        "validation_percent",
        "validation_seed",
    ]

    for key in critical_keys:
        if saved_config.get(key) != current_config.get(key):
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAE on representations from SD layer using RepresentationDataset."
    )
    # Dataset parameters - support multiple paths
    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to dataset directories containing layer representations "
        "(e.g., results/sd_1_5/dataset/representations/unet_up_1_att_1). "
        "Multiple paths will be combined.",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        required=False,
        default=None,
        help="Path to separate test dataset directory. Optional. "
        "If not provided, use --validation_percent for internal split.",
    )
    parser.add_argument(
        "--datasets_name",
        type=str,
        required=True,
        help="Name for the combined datasets (used for logging/organization)",
    )

    # Validation split parameters
    parser.add_argument(
        "--validation_percent",
        type=int,
        default=0,
        help="Percentage of data to use for validation (0-100). "
        "Set to 0 to disable validation split.",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val splits",
    )

    # SAE parameters
    parser.add_argument(
        "--sae_path",
        type=str,
        required=True,
        help="Path to .pt file for SAE weights (load if exists, create and save if not)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to save SAE config JSON",
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

    # Learning rate scheduler parameters
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of batches for linear warmup phase. "
        "Set to 0 to disable scheduler. Default: 0 (disabled)",
    )
    parser.add_argument(
        "--warmup_start_factor",
        type=float,
        default=0.01,
        help="Starting LR as fraction of base LR during warmup. "
        "E.g., 0.01 means start at 1%% of learning_rate. Default: 0.01",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.0,
        help="Minimum LR as fraction of base LR after cosine decay. "
        "E.g., 0.1 means decay to 10%% of learning_rate. Default: 0.0",
    )

    parser.add_argument("--skip-wandb", action="store_true")
    return parser.parse_args()


def load_dataset_from_path(dataset_path: str) -> RepresentationDataset:
    """
    Load a RepresentationDataset from a path.

    The path should be to a layer directory containing data.npy and metadata.json,
    e.g., /path/to/results/sd_v1_5/cc3m-wds/representations/unet_up_1_att_1

    We extract cache_dir and layer_name from this path structure.
    """
    path = Path(dataset_path)

    # The path IS the layer directory (contains data.npy, metadata.json)
    # So cache_dir = parent, layer_name = name
    layer_name = path.name
    cache_dir = path.parent

    print(f"  Loading dataset: cache_dir={cache_dir}, layer={layer_name}")

    return RepresentationDataset(
        cache_dir=cache_dir,
        layer_name=layer_name,
        return_metadata=False,
    )


class ShuffledRangeSampler(Sampler[int]):
    """
    Memory-efficient sampler that regenerates indices from seed.

    This sampler does NOT store any index arrays. Instead, it regenerates
    the shuffled indices on-the-fly using the seed.
    """

    def __init__(
        self,
        total_size: int,
        start_idx: int,
        end_idx: int,
        shuffle: bool = True,
        seed: int = 42,
        chunk_size: int = 1_000_000,
    ):
        """
        Args:
            total_size: Total dataset size (for permutation generation)
            start_idx: Start index in the shuffled order (inclusive)
            end_idx: End index in the shuffled order (exclusive)
            shuffle: Whether to shuffle iteration order each epoch
            seed: Random seed for reproducible shuffling
            chunk_size: Chunk size for memory-efficient iteration
        """
        self._total_size = total_size
        self._start = start_idx
        self._end = end_idx
        self._length = end_idx - start_idx
        self._shuffle = shuffle
        self._seed = seed
        self._chunk_size = chunk_size
        self.epoch = 0

    def _get_indices_chunk(self, chunk_start: int, chunk_end: int) -> np.ndarray:
        """
        Get a chunk of the shuffled indices by regenerating from seed.

        Args:
            chunk_start: Start index of the chunk (relative to sampler)
            chunk_end: End index of the chunk (relative to sampler)

        Returns:
            Numpy array of indices for the specified chunk

        """

        # Use the same seed every time to get consistent permutation
        rng = np.random.default_rng(self._seed)
        # Generate full permutation
        full_perm = rng.permutation(self._total_size)
        # Return only our chunk from within our assigned range
        return full_perm[self._start + chunk_start : self._start + chunk_end]

    def __iter__(self) -> Iterator[int]:
        # We need to iterate over indices [_start, _end) from the master permutation
        # But we don't want to store the full permutation

        # Strategy: Generate full permutation once per epoch, iterate in chunks
        rng_master = np.random.default_rng(self._seed)
        full_perm = rng_master.permutation(self._total_size)

        # Extract our slice
        my_indices = full_perm[self._start : self._end]
        del full_perm

        if self._shuffle:
            # Shuffle our slice for this epoch
            rng_epoch = np.random.default_rng(self._seed + self.epoch + 1000)
            n_chunks = (self._length + self._chunk_size - 1) // self._chunk_size
            chunk_order = rng_epoch.permutation(n_chunks)

            for chunk_idx in chunk_order:
                chunk_start = chunk_idx * self._chunk_size
                chunk_end = min(chunk_start + self._chunk_size, self._length)

                chunk = my_indices[chunk_start:chunk_end].copy()
                rng_epoch.shuffle(chunk)

                for idx in chunk:
                    yield int(idx)

                del chunk
        else:
            # No shuffle - iterate in order through our slice
            for idx in my_indices:
                yield int(idx)

        del my_indices

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible per-epoch shuffling."""
        self.epoch = epoch


def create_train_val_split(dataset, val_percent: int, seed: int):
    """
    Split a dataset into train and validation subsets.

    Args:
        dataset: The dataset to split
        val_percent: Percentage for validation (0-100)
        seed: Random seed for reproducibility

    Returns:
        (dataset, train_sampler, val_sampler) - samplers are None if val_percent is 0
    """
    if val_percent <= 0 or val_percent >= 100:
        return dataset, None, None

    n_samples = len(dataset)
    n_val = int(n_samples * val_percent / 100)
    n_train = n_samples - n_val

    # Create stateless samplers - they regenerate indices from seed
    # Memory per sampler: ~100 bytes (just stores integers)
    # Each worker regenerates the permutation independently
    train_sampler = ShuffledRangeSampler(
        total_size=n_samples,
        start_idx=0,
        end_idx=n_train,
        shuffle=True,
        seed=seed,
    )
    val_sampler = ShuffledRangeSampler(
        total_size=n_samples,
        start_idx=n_train,
        end_idx=n_samples,
        shuffle=False,
        seed=seed,
    )

    print(f"  Split: {n_train:,} train, {n_val:,} validation samples")

    return dataset, train_sampler, val_sampler


def main() -> int:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("LOAD DATASET AND TRAIN SAE ON REPRESENTATIONS")
    print("=" * 80)
    print(f"Dataset paths: {args.dataset_paths}")
    print(f"Datasets name: {args.datasets_name}")
    print(f"Test dataset path: {args.test_dataset_path}")
    print(f"Validation percent: {args.validation_percent}%")
    print(f"Validation seed: {args.validation_seed}")
    print(f"SAE path: {args.sae_path}")
    print(f"Config path: {args.config_path}")
    print(f"Device: {device}")
    print(
        f"SAE Config: epochs={args.num_epochs}, lr={args.learning_rate}, "
        f"exp_factor={args.expansion_factor}, top_k={args.top_k}, batch_size={args.batch_size}"
    )
    print("-" * 80)

    try:
        # Extract layer name from first dataset path for naming
        layer_name = Path(args.dataset_paths[0]).name

        # Build extended run name with hyperparameters
        lr_str = f"{args.learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        run_name = (
            f"SAE_{args.datasets_name}_{layer_name}_"
            f"exp{args.expansion_factor}_k{args.top_k}_lr{lr_str}_"
            f"ep{args.num_epochs}_bs{args.batch_size}"
        )

        # Build comprehensive tags
        tags = [
            "sae",
            "training",
            f"layer:{layer_name}",
            f"exp:{args.expansion_factor}",
            f"topk:{args.top_k}",
            f"lr:{lr_str}",
            f"dataset:{args.datasets_name}",
        ]

        # Add SLURM info to tags if available
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_job_id:
            tags.append(f"slurm:{slurm_job_id}")
        if slurm_array_task_id:
            tags.append(f"array:{slurm_array_task_id}")

        # Add scheduler tag if enabled
        if args.warmup_steps > 0:
            tags.append("scheduler:warmup_cosine")
            tags.append(f"warmup:{args.warmup_steps}")

        # Initialize wandb
        if not args.skip_wandb:
            wandb.login()
            wandb.init(
                project="sd-control-representation",
                entity="bartoszjezierski28-warsaw-university-of-technology",
                name=run_name,
                config={
                    # Dataset config
                    "datasets_name": args.datasets_name,
                    "dataset_paths": args.dataset_paths,
                    "layer_name": layer_name,
                    "val_dataset_path": args.test_dataset_path,
                    "validation_percent": args.validation_percent,
                    "validation_seed": args.validation_seed,
                    # Model config
                    "expansion_factor": args.expansion_factor,
                    "top_k": args.top_k,
                    "input_dim": None,  # Will be updated after loading dataset
                    "nb_concepts": None,  # Will be updated after loading dataset
                    # Training config
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    # Scheduler config
                    "scheduler_enabled": args.warmup_steps > 0,
                    "scheduler_warmup_steps": args.warmup_steps,
                    "scheduler_warmup_start_factor": args.warmup_start_factor,
                    "scheduler_min_lr_ratio": args.min_lr_ratio,
                    # Paths
                    "sae_path": args.sae_path,
                    "config_path": args.config_path,
                    # System info
                    "device": device,
                    "slurm_job_id": slurm_job_id,
                    "slurm_array_task_id": slurm_array_task_id,
                    "hostname": os.environ.get("HOSTNAME", "unknown"),
                },
                tags=tags,
                notes=f"Training SAE on {args.datasets_name} ({layer_name}) "
                f"with exp={args.expansion_factor}, k={args.top_k}, lr={lr_str}",
            )

            # Define separate x-axes for batch and epoch metrics
            # This prevents batch step numbers from affecting epoch metrics
            wandb.define_metric("batch/*", step_metric="batch/step")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("gap/*", step_metric="epoch")
            wandb.define_metric("hparams/*", step_metric="epoch")
            wandb.define_metric("meta/*", step_metric="epoch")

        # Load all datasets
        print("\nLoading datasets...")
        datasets = []
        input_dim = None

        for path in args.dataset_paths:
            ds = load_dataset_from_path(path)
            datasets.append(ds)

            # Verify all datasets have the same feature dimension
            if input_dim is None:
                input_dim = ds.feature_dim
            elif ds.feature_dim != input_dim:
                raise ValueError(
                    f"Feature dimension mismatch: expected {input_dim}, "
                    f"got {ds.feature_dim} for {path}"
                )

        # Combine datasets if multiple
        if len(datasets) == 1:
            combined_dataset = datasets[0]
        else:
            combined_dataset = ConcatDataset(datasets)
            print(f"\nCombined {len(datasets)} datasets: {len(combined_dataset)} total samples")

        print(f"Total samples: {len(combined_dataset)}, input_dim={input_dim}")

        # Update wandb config with input_dim and nb_concepts
        nb_concepts = input_dim * args.expansion_factor
        if not args.skip_wandb:
            wandb.config.update(
                {
                    "input_dim": input_dim,
                    "nb_concepts": nb_concepts,
                    "total_samples": len(combined_dataset),
                },
                allow_val_change=True,
            )

        # Handle validation dataset
        validation_dataset = None
        training_dataset = combined_dataset
        train_sampler = None
        val_sampler = None

        if args.test_dataset_path:
            # Use separate validation dataset
            print(f"\nLoading separate validation dataset from: {args.test_dataset_path}")
            validation_dataset = load_dataset_from_path(args.test_dataset_path)
            print(f"Validation dataset: {len(validation_dataset)} samples")
        elif args.validation_percent > 0:
            # Create internal train/val split with memory-efficient samplers
            print(
                f"\nCreating {args.validation_percent}% validation split "
                f"(seed={args.validation_seed})..."
            )
            training_dataset, train_sampler, val_sampler = create_train_val_split(
                combined_dataset, args.validation_percent, args.validation_seed
            )
            # training_dataset is the original combined_dataset (no Subset wrapping)
            # samplers handle the index subsetting
        else:
            print("\nNo validation dataset - training without validation")

        # Calculate optimal DataLoader parameters to maximize GPU utilization
        is_cuda = device == "cuda"
        dl_params = get_optimal_dataloader_params(is_cuda=is_cuda)

        print("\nDataLoader configuration:")
        print(f"  num_workers: {dl_params['num_workers']}")
        print(f"  prefetch_factor: {dl_params['prefetch_factor']}")
        print(f"  pin_memory: {dl_params['pin_memory']}")
        print(f"  persistent_workers: {dl_params['persistent_workers']}")
        if os.environ.get("SLURM_CPUS_PER_TASK"):
            print(f"  (SLURM_CPUS_PER_TASK={os.environ['SLURM_CPUS_PER_TASK']})")

        # Prepare DataLoaders
        # Use sampler if available (memory-efficient for large datasets)
        # When using sampler, shuffle must be False (sampler handles shuffling)
        if train_sampler is not None:
            train_dataloader = DataLoader(
                training_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                **dl_params,
            )
        else:
            train_dataloader = DataLoader(
                training_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                **dl_params,
            )

        # Create validation dataloader only if validation sampler or dataset exists
        val_dataloader = None
        if val_sampler is not None:
            # Using sampler-based validation (from internal split)
            val_dataloader = DataLoader(
                training_dataset,  # Same dataset, different sampler
                batch_size=args.batch_size,
                sampler=val_sampler,
                **dl_params,
            )
            print(f"Validation dataloader: {len(val_dataloader)} batches")
        elif validation_dataset is not None:
            # Using separate validation dataset
            val_dataloader = DataLoader(
                validation_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                **dl_params,
            )
            print(f"Validation dataloader: {len(val_dataloader)} batches")

        print(f"Training dataloader: {len(train_dataloader)} batches")

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
        config_path = Path(args.config_path)
        checkpoint_path = get_checkpoint_path(sae_path)

        # Build current config for comparison
        current_config = {
            "input_dim": input_dim,
            "nb_concepts": nb_concepts,
            "expansion_factor": args.expansion_factor,
            "top_k": args.top_k,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "datasets_name": args.datasets_name,
            "dataset_paths": args.dataset_paths,
            "validation_percent": args.validation_percent,
            "validation_seed": args.validation_seed,
            # Scheduler configuration
            "scheduler": {
                "enabled": args.warmup_steps > 0,
                "warmup_steps": args.warmup_steps,
                "warmup_start_factor": args.warmup_start_factor,
                "min_lr_ratio": args.min_lr_ratio,
            },
        }

        # Define optimizer
        optimizer = torch.optim.Adam(sae.parameters(), lr=args.learning_rate)

        # Try to resume from checkpoint
        start_epoch = 0
        existing_logs = None

        checkpoint, start_epoch, existing_logs = load_checkpoint(checkpoint_path, device)
        if checkpoint is not None:
            saved_config = checkpoint.get("config", {})
            if check_config_match(saved_config, current_config):
                # Valid checkpoint - resume training
                sae.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print(f"  ✓ Resuming training from epoch {start_epoch + 1}")
            else:
                print("  ⚠ Checkpoint config mismatch - starting fresh")
                start_epoch = 0
                existing_logs = None
        elif sae_path.exists():
            # No checkpoint but model exists - load it (might be final model)
            print(f"Loading existing SAE from: {sae_path}")
            sae.load_state_dict(torch.load(sae_path, map_location=device))
        else:
            print(f"Creating new SAE (file does not exist: {sae_path})")

        # Check if training is already complete
        if start_epoch >= args.num_epochs:
            print(f"\n✓ Training already complete ({args.num_epochs} epochs)")
            if existing_logs is not None:
                results_logs = existing_logs
            else:
                print("  Warning: No logs found for completed training")
                results_logs = {"train": {}, "val": {}}
        else:
            # Create scheduler config
            scheduler_config = SchedulerConfig(
                enabled=args.warmup_steps > 0,
                warmup_steps=args.warmup_steps,
                warmup_start_factor=args.warmup_start_factor,
                min_lr_ratio=args.min_lr_ratio,
            )

            # Calculate total training steps for scheduler
            total_steps = len(train_dataloader) * (args.num_epochs - start_epoch)

            # Create learning rate scheduler (if enabled)
            scheduler = create_warmup_cosine_scheduler(
                optimizer=optimizer,
                scheduler_config=scheduler_config,
                total_steps=total_steps,
            )

            if scheduler is not None:
                actual_warmup = (
                    min(args.warmup_steps, total_steps - 1) if args.warmup_steps > 0 else 0
                )
                print("\n📈 Learning rate scheduler:")
                if actual_warmup < args.warmup_steps:
                    print(f"   Warmup: {actual_warmup} steps (clamped from {args.warmup_steps})")
                else:
                    print(f"   Warmup: {args.warmup_steps} steps")
                print(f"   Start factor: {args.warmup_start_factor:.2%} of base LR")
                print(f"   Min LR ratio: {args.min_lr_ratio:.2%} of base LR")
                print(f"   Total steps: {total_steps}")

            # Create training config
            training_config = TrainingConfig(
                nb_epochs=args.num_epochs,
                clip_grad=1.0,
                use_amp=True,
                log_interval=10,
                device=device,
                start_epoch=start_epoch,
            )

            # Create trainer
            trainer = SAETrainer(
                model=sae,
                optimizer=optimizer,
                config=training_config,
                scheduler=scheduler,
                base_lr=args.learning_rate,
            )

            # Add batch callback for WandB logging (logs every N batches)
            if not args.skip_wandb:
                batch_log_interval = 10  # Log every N batches

                def wandb_batch_callback(step, batch_idx, epoch, metrics):
                    """Log batch metrics to WandB."""
                    # Only log every N batches to avoid overwhelming WandB
                    if batch_idx % batch_log_interval != 0:
                        return

                    # Filter out None and infinite values
                    log_dict = {"batch/step": step, "batch/epoch": epoch}

                    metric_mappings = {
                        "batch/loss": "loss",
                        "batch/recon_loss": "recon_loss",
                        "batch/aux_loss": "aux_loss",
                        "batch/r2": "r2",
                        "batch/l0_sparsity": "l0_sparsity",
                        "batch/z_l2": "z_l2",
                        "batch/mean_activation": "mean_activation",
                        "batch/active_ratio": "active_ratio",
                    }

                    for wandb_key, metric_key in metric_mappings.items():
                        value = metrics.get(metric_key)
                        if value is not None and not (
                            isinstance(value, float)
                            and (value != value or abs(value) == float("inf"))
                        ):
                            log_dict[wandb_key] = value

                    # Add learning rate
                    if scheduler is not None:
                        try:
                            log_dict["batch/learning_rate"] = scheduler.get_last_lr()[0]
                        except Exception:
                            pass

                    wandb.log(log_dict, step=step)

                trainer.add_batch_callback(wandb_batch_callback)

            # Add checkpoint callback
            def checkpoint_callback(
                model, optimizer, epoch, train_metrics, val_metrics, train_logs, val_logs
            ):
                _ = train_metrics, val_metrics  # Unused, but required
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_logs=train_logs,
                    val_logs=val_logs,
                    config=current_config,
                )

            trainer.add_epoch_callback(checkpoint_callback)

            # Train SAE
            print("\nStarting SAE training...")
            results_logs = trainer.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                existing_logs=existing_logs,
            )
            print("\n✓ SAE training completed.")

        sae = sae.eval()

        # Log to wandb
        if not args.skip_wandb and results_logs and "train" in results_logs:
            system_metrics_end = get_system_metrics(device)

            # Get number of logged epochs from actual logs
            num_logged_epochs = len(results_logs["train"].get("avg_loss", []))

            for epoch in range(num_logged_epochs):
                train_logs = results_logs["train"]

                # Core training metrics
                log_dict = {
                    "epoch": epoch + 1,
                    # Learning rate (actual LR at this epoch, important for scheduler)
                    "train/learning_rate": train_logs.get(
                        "learning_rate", [args.learning_rate] * num_logged_epochs
                    )[epoch],
                    # Training metrics - loss components
                    "train/loss": train_logs["avg_loss"][epoch],
                    "train/recon_loss": train_logs.get("recon_loss", [0] * num_logged_epochs)[
                        epoch
                    ],
                    "train/aux_loss": train_logs.get("aux_loss", [0] * num_logged_epochs)[epoch],
                    "train/r2_score": train_logs["r2"][epoch],
                    "train/l0_sparsity": train_logs["z_sparsity"][epoch],
                    "train/z_l2": train_logs.get("z_l2", [0] * num_logged_epochs)[epoch],
                    "train/dead_features_ratio": train_logs["dead_features"][epoch],
                    "train/dead_features_pct": train_logs["dead_features"][epoch] * 100,
                    "train/epoch_time_s": train_logs["time_epoch"][epoch],
                    # Timing metrics
                    "train/data_loading_time_s": train_logs.get(
                        "data_loading_time", [0] * num_logged_epochs
                    )[epoch],
                    "train/batch_compute_time_s": train_logs.get(
                        "batch_compute_time", [0] * num_logged_epochs
                    )[epoch],
                    "train/avg_data_loading_time_ms": train_logs.get(
                        "avg_data_loading_time_ms", [0] * num_logged_epochs
                    )[epoch],
                    "train/avg_batch_compute_time_ms": train_logs.get(
                        "avg_batch_compute_time_ms", [0] * num_logged_epochs
                    )[epoch],
                    # Dictionary metrics
                    "train/dict_sparsity": train_logs.get("dict_sparsity", [0] * num_logged_epochs)[
                        epoch
                    ],
                    "train/dict_norms_mean": train_logs.get(
                        "dict_norms_mean", [0] * num_logged_epochs
                    )[epoch],
                    # Hyperparameters (for filtering in wandb UI)
                    "hparams/expansion_factor": args.expansion_factor,
                    "hparams/top_k": args.top_k,
                    "hparams/learning_rate": args.learning_rate,
                    "hparams/batch_size": args.batch_size,
                    # Identifiers
                    "meta/datasets_name": args.datasets_name,
                    "meta/layer_name": layer_name,
                }

                # Add validation metrics if available
                if "val" in results_logs and results_logs["val"] is not None:
                    val_logs = results_logs["val"]
                    if epoch < len(val_logs.get("avg_loss", [])):
                        log_dict.update(
                            {
                                # Loss components
                                "val/loss": val_logs["avg_loss"][epoch],
                                "val/recon_loss": val_logs.get(
                                    "recon_loss", [0] * num_logged_epochs
                                )[epoch],
                                "val/aux_loss": val_logs.get("aux_loss", [0] * num_logged_epochs)[
                                    epoch
                                ],
                                "val/r2_score": val_logs["r2"][epoch],
                                "val/l0_sparsity": val_logs["z_sparsity"][epoch],
                                "val/z_l2": val_logs.get("z_l2", [0] * num_logged_epochs)[epoch],
                                "val/dead_features_ratio": val_logs["dead_features"][epoch],
                                "val/dead_features_pct": val_logs["dead_features"][epoch] * 100,
                                # Timing metrics
                                "val/data_loading_time_s": val_logs.get(
                                    "data_loading_time", [0] * num_logged_epochs
                                )[epoch],
                                "val/batch_compute_time_s": val_logs.get(
                                    "batch_compute_time", [0] * num_logged_epochs
                                )[epoch],
                                "val/avg_data_loading_time_ms": val_logs.get(
                                    "avg_data_loading_time_ms", [0] * num_logged_epochs
                                )[epoch],
                                "val/avg_batch_compute_time_ms": val_logs.get(
                                    "avg_batch_compute_time_ms", [0] * num_logged_epochs
                                )[epoch],
                                # Similarity metrics
                                "val/encoder_avg_max_cos": val_logs.get(
                                    "encoder_avg_max_cos", [0] * num_logged_epochs
                                )[epoch],
                                "val/decoder_avg_max_cos": val_logs.get(
                                    "decoder_avg_max_cos", [0] * num_logged_epochs
                                )[epoch],
                                "val/decoder_mean_norm": val_logs.get(
                                    "decoder_mean_norm", [0] * num_logged_epochs
                                )[epoch],
                                # Active features at thresholds
                                "val/active_features_0.5": val_logs.get(
                                    "active_features_0.5", [0] * num_logged_epochs
                                )[epoch],
                                "val/active_features_0.4": val_logs.get(
                                    "active_features_0.4", [0] * num_logged_epochs
                                )[epoch],
                                "val/active_features_0.3": val_logs.get(
                                    "active_features_0.3", [0] * num_logged_epochs
                                )[epoch],
                                "val/active_features_0.2": val_logs.get(
                                    "active_features_0.2", [0] * num_logged_epochs
                                )[epoch],
                                "val/active_features_0.1": val_logs.get(
                                    "active_features_0.1", [0] * num_logged_epochs
                                )[epoch],
                                # Gap metrics (for overfitting detection)
                                "gap/loss": train_logs["avg_loss"][epoch]
                                - val_logs["avg_loss"][epoch],
                                "gap/r2": train_logs["r2"][epoch] - val_logs["r2"][epoch],
                            }
                        )

                log_dict.update(system_metrics_end)
                wandb.log(log_dict)

            # Log final summary metrics
            train_logs = results_logs["train"]
            if wandb.run is not None:
                wandb.run.summary.update(
                    {
                        "final/train_loss": train_logs["avg_loss"][-1],
                        "final/train_r2": train_logs["r2"][-1],
                        "final/train_dead_pct": train_logs["dead_features"][-1] * 100,
                        "final/total_epochs": num_logged_epochs,
                        "final/total_time_s": sum(train_logs["time_epoch"]),
                        "final/avg_epoch_time_s": sum(train_logs["time_epoch"]) / num_logged_epochs,
                        "final/total_data_loading_time_s": sum(
                            train_logs.get("data_loading_time", [0])
                        ),
                        "final/total_batch_compute_time_s": sum(
                            train_logs.get("batch_compute_time", [0])
                        ),
                    }
                )

                if "val" in results_logs and results_logs["val"] is not None:
                    val_logs = results_logs["val"]
                    wandb.run.summary.update(
                        {
                            "final/val_loss": val_logs["avg_loss"][-1],
                            "final/val_r2": val_logs["r2"][-1],
                            "final/val_dead_pct": val_logs["dead_features"][-1] * 100,
                            "final/best_val_loss": min(val_logs["avg_loss"]),
                            "final/best_val_r2": max(val_logs["r2"]),
                            "final/val_encoder_avg_max_cos": val_logs.get(
                                "encoder_avg_max_cos", [0]
                            )[-1],
                            "final/val_decoder_avg_max_cos": val_logs.get(
                                "decoder_avg_max_cos", [0]
                            )[-1],
                            "final/val_active_features_0.1": val_logs.get(
                                "active_features_0.1", [0]
                            )[-1],
                        }
                    )

        # Save the trained model and config
        sae_path.parent.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print("SAVING MODEL AND CONFIG")
        print("=" * 80)

        # Save model
        print(f"Saving trained SAE model to: {sae_path}")
        torch.save(sae.state_dict(), sae_path)
        print("✓ Model saved successfully")

        # Save config (use current_config defined earlier)
        print(f"Saving config to: {config_path}")
        with open(config_path, "w") as f:
            json.dump(current_config, f, indent=2)
        print("✓ Config saved successfully")

        # Remove checkpoint after successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("✓ Checkpoint removed (training complete)")

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
