#!/usr/bin/env python3
"""
List all leaf files in cached_rep/obj_*/style_* directories, load them into a single tensor,
and train a Sparse Autoencoder (SAE) on the concatenated data.

Traverses the directory tree:
    {stable_diffusion_model_path}/{layer_in_model_path}/cached_rep/obj_[1-3]/style_[1-3]/

Loads .pt files with shape [timesteps, num_of_obs, dimension] and concatenates them
into a single tensor of shape [total_num_of_obs_for_every_timestep, dimension].
Then trains an SAE on this data.
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import sys
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader, TensorDataset
from overcomplete.sae import TopKSAE, train_sae
from src.models.sae.training import criterion_laux

# Add project root to path to allow imports to work from any location
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env from project root
load_dotenv(dotenv_path=project_root / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load cached representations and train SAE on concatenated tensor."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Base path to Stable Diffusion model outputs (e.g., results/enum_layer_1)",
    )
    parser.add_argument(
        "--layer_name",
        type=str,
        required=True,
        help="Layer path (e.g., SAE or UNET_UP_1_ATT_2)",
    )
    parser.add_argument(
        "--num_of_epochs",
        default=5,
        type=int,
        required=True,
        help="Number of SAE training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        required=True,
        help="SAE learning rate",
    )
    parser.add_argument(
        "--expansion_factor",
        type=int,
        default=16,
        required=True,
        help="SAE expansion factor",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=32,
        required=True,
        help="SAE top-k sparsity",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for SAE training",
    )
    return parser.parse_args()


def find_leaf_files(base_path: Path) -> List[Path]:
    """
    Find all files in leaf directories: cached_rep/obj_*/style_*/*
    """
    leaf_files = []
    pattern = base_path / "cached_rep" / "obj_*" / "style_*"

    for style_dir in sorted(pattern.glob("*")):
        if not style_dir.is_dir():
            continue
        for file_path in sorted(style_dir.iterdir()):
            if file_path.is_file() and file_path.suffix == ".pt":
                leaf_files.append(file_path)
    return leaf_files


def load_and_concatenate_tensors(file_paths: List[Path]) -> Tuple[torch.Tensor, dict]:
    """
    Load all .pt files and concatenate them into a single tensor.

    Expected input shape: [timesteps, num_of_obs, dimension]
    Output shape: [total_num_of_obs_for_every_timestep, dimension]

    Returns:
        concatenated_tensor: The concatenated tensor
        metadata: Dictionary with information about the loading process
    """
    if not file_paths:
        raise ValueError("No files to load")

    all_tensors = []
    metadata = {
        "num_files": len(file_paths),
        "file_shapes": [],
        "total_obs": 0,
        "dimension": None,
        "num_timesteps": None,
    }

    print("\nLoading tensors...")
    for i, file_path in enumerate(file_paths, 1):
        print(f"  [{i}/{len(file_paths)}] Loading {file_path.name}...", end=" ")

        try:
            tensor = torch.load(file_path, map_location="cpu")

            # Validate shape: should be [timesteps, num_of_obs, dimension]
            if tensor.dim() != 3:
                raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D: {tensor.shape}")

            timesteps, num_obs, dimension = tensor.shape
            metadata["file_shapes"].append(tensor.shape)

            # Check consistency across files
            if metadata["dimension"] is None:
                metadata["dimension"] = dimension
                metadata["num_timesteps"] = timesteps
            else:
                if dimension != metadata["dimension"]:
                    raise ValueError(
                        f"Dimension mismatch: expected {metadata['dimension']}, got {dimension}"
                    )
                if timesteps != metadata["num_timesteps"]:
                    raise ValueError(
                        f"Timesteps mismatch: expected {metadata['num_timesteps']}, got {timesteps}"
                    )

            # Reshape: [timesteps, num_obs, dimension] -> [timesteps * num_obs, dimension]
            reshaped = tensor.reshape(-1, dimension)
            all_tensors.append(reshaped)
            metadata["total_obs"] += timesteps * num_obs

            print(f"✓ Shape: {tensor.shape} -> {reshaped.shape}")

        except Exception as e:
            print(f"✗ ERROR: {e}")
            raise

    # Concatenate all tensors along the observation dimension
    print("\nConcatenating tensors...")
    concatenated = torch.cat(all_tensors, dim=0)

    print(f"✓ Final shape: {concatenated.shape}")
    print(f"  Total observations: {metadata['total_obs']:,}")
    print(f"  Dimension: {metadata['dimension']}")

    return concatenated, metadata


def train_sae_model(
    activations: torch.Tensor,
    input_dim: int,
    expansion_factor: int,
    top_k: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    device: torch.device,
) -> TopKSAE:
    """
    Train a Sparse Autoencoder on the given activations.

    Args:
        activations: Input tensor of shape (num_samples, input_dim)
        input_dim: Dimension of input features
        expansion_factor: SAE expansion factor
        top_k: Number of active units (sparsity)
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on

    Returns:
        Trained SAE model
    """
    print("\n" + "=" * 80)
    print("SAE TRAINING")
    print("=" * 80)
    print(f"Input dimension: {input_dim}")
    print(f"Number of concepts: {input_dim * expansion_factor}")
    print(f"Expansion factor: {expansion_factor}")
    print(f"Top-k sparsity: {top_k}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("-" * 80)

    # Move data to device
    activations = activations.to(device)

    # Prepare DataLoader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize SAE model
    nb_concepts = input_dim * expansion_factor
    sae = TopKSAE(input_dim, nb_concepts=nb_concepts, top_k=top_k, device=device)
    sae = sae.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

    # Train SAE
    print("\nStarting SAE training...")
    train_sae(sae, dataloader, criterion_laux, optimizer, nb_epochs=num_epochs, device=device)
    sae = sae.eval()
    print("\n✓ SAE training completed.")

    return sae


def main() -> int:
    args = parse_args()

    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get results directory from environment variable or use default
    results_base_dir = os.environ.get("RESULTS_DIR")
    if results_base_dir:
        results_base_path = Path(results_base_dir)
    else:
        # Default to relative path from project root
        results_base_path = Path(__file__).parent.parent.parent / "results"

    # Construct full path
    root_path = results_base_path / args.stable_diffusion_model_path / args.layer_in_model_path
    cached_rep_path = root_path / "cached_rep"

    if not cached_rep_path.exists():
        print(f"Error: Path does not exist: {cached_rep_path}")
        return 1

    print("=" * 80)
    print("LOAD, CONCATENATE AND TRAIN SAE ON CACHED REPRESENTATIONS")
    print("=" * 80)
    print(f"Root: {root_path}")
    print(f"Layer: {args.layer_in_model_path}")
    print(f"Device: {device}")
    print(
        f"SAE Config: epochs={args.num_of_epochs}, lr={args.learning_rate}, "
        f"exp_factor={args.expansion_factor}, top_k={args.top_k}"
    )
    print("-" * 80)

    # Find all .pt files
    leaf_files = find_leaf_files(root_path)

    if not leaf_files:
        print("No .pt files found. Check the directory structure.")
        print("Expected: .../cached_rep/obj_[1-3]/style_[1-3]/<files>.pt")
        return 1

    print(f"Found {len(leaf_files)} .pt file(s) in leaf directories:\n")
    for i, file_path in enumerate(leaf_files, 1):
        rel_path = file_path.relative_to(root_path)
        print(f"[{i:3d}] {rel_path}  ({file_path.stat().st_size // 1024:,} KB)")

    print("\n" + "-" * 80)

    # Load and concatenate tensors
    try:
        concatenated_tensor, metadata = load_and_concatenate_tensors(leaf_files)

        print("\n" + "=" * 80)
        print("DATA SUMMARY")
        print("=" * 80)
        print(f"Files processed: {metadata['num_files']}")
        print(f"Final tensor shape: {concatenated_tensor.shape}")
        print(f"Total observations: {metadata['total_obs']:,}")
        print(f"Dimension: {metadata['dimension']}")
        print(
            f"Memory usage: {concatenated_tensor.element_size() * concatenated_tensor.nelement() / (1024**2):.2f} MB"
        )

        # Train SAE on the concatenated data
        sae = train_sae_model(
            activations=concatenated_tensor,
            input_dim=metadata["dimension"],
            expansion_factor=args.expansion_factor,
            top_k=args.top_k,
            learning_rate=args.learning_rate,
            num_epochs=args.num_of_epochs,
            batch_size=args.batch_size,
            device=device,
        )

        # Save the trained model
        model_path = root_path / "SAE"
        learning_rate_str = f"{args.learning_rate:.5e}".replace("-", "m").replace("+", "p")
        model_path /= f"sae_exp{args.expansion_factor}_topk{args.top_k}_lr{learning_rate_str}_epochs{args.num_of_epochs}_batch{args.batch_size}.pt"

        print("\n" + "=" * 80)
        print("SAVING MODEL")
        print("=" * 80)
        print(f"Saving trained SAE model to: {model_path}")
        torch.save(sae.state_dict(), model_path)
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
