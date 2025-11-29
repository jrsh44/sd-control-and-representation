#!/usr/bin/env python3

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
from src.models.sae.feature_selection import (  # noqa: E402
    compute_sums,
    concept_filtering_function,
    infer_sae_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAE on representations from SD layer using RepresentationDataset."
    )
    # Dataset parameters
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to training dataset directory (e.g., results/sd_1_5/unet_mid_att)",
    )
    # concept
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        help="Concept name for feature selection",
    )
    # concept value
    parser.add_argument(
        "--concept_value",
        type=str,
        required=True,
        help="Concept value for feature selection",
    )
    # SAE path
    parser.add_argument(
        "--sae_path",
        type=str,
        required=True,
        help="Path to .pt file for SAE weights (load if exists, create and save if not)",
    )
    # feature score path
    parser.add_argument(
        "--feature_scores_path",
        type=str,
        required=True,
        help="Path to .npy file for feature scores (load if exists, create and save if not)",
    )
    # epsilon
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        required=False,
        help="Small value to avoid division by zero",
    )
    # dataLoader batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        required=False,
        help="Batch size for DataLoader",
    )
    # skip wandb
    parser.add_argument("--skip-wandb", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("LOAD DATASETS AND SELECT FEATURES USING PRETRAINED SAE GIVEN CONCEPT")
    print("=" * 80)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Concept: {args.concept}")
    print(f"Concept value: {args.concept_value}")
    print(f"SAE path: {args.sae_path}")
    print(f"Feature scores path: {args.feature_scores_path}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Device: {device}")
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
                    "dataset_path": args.dataset_path,
                    "concept": args.concept,
                    "concept_value": args.concept_value,
                    "epsilon": args.epsilon,
                    "batch_size": args.batch_size,
                },
                tags=["sae", "feature_selection"],
                notes="Feature selection using pretrained SAE",
            )

        # Load SAE → infer config
        sae_path = Path(args.sae_path)
        if not sae_path.exists():
            raise FileNotFoundError(f"SAE not found: {sae_path}")

        print(f"Loading SAE from {sae_path}...")
        state_dict = torch.load(sae_path, map_location="cpu")

        # Create dummy model to infer shapes
        dummy_sae = TopKSAE(input_dim=1, nb_concepts=1, top_k=1, device=device)
        dummy_sae.load_state_dict(state_dict, strict=False)  # partial load to get shapes

        config = infer_sae_config(dummy_sae)
        print("Inferred SAE config:")
        for k, v in config.items():
            print(f"  {k}: {v}")

        # Re-create the *real* model with correct dims
        sae = TopKSAE(
            input_dim=config["input_dim"],
            nb_concepts=config["nb_concepts"],
            top_k=config["top_k"],
            device=device,
        )
        sae.load_state_dict(state_dict)  # now full load
        sae = sae.to(device)
        sae.eval()
        print("SAE loaded and moved to device")

        print("Compiling model with torch.compile() (first epoch will be slower)...")
        sae = torch.compile(sae)
        print("✓ Model compiled successfully")

        # Load dataset
        dataset_concept_true = RepresentationDataset(
            dataset_path=args.dataset_path,
            flatten=True,  # you already flatten
            filter_fn=concept_filtering_function(args.concept, args.concept_value),
            return_metadata=False,
        )
        n_samples_true = dataset_concept_true.feature_dim
        print(f"Loaded dataset: {len(dataset_concept_true)} samples, input_dim={n_samples_true}")

        dataset_concept_false = RepresentationDataset(
            dataset_path=args.dataset_path,
            flatten=True,  # you already flatten
            filter_fn=concept_filtering_function(args.concept, args.concept_value, negate=True),
            return_metadata=False,
        )
        n_samples_false = dataset_concept_false.feature_dim
        print(f"Loaded dataset: {len(dataset_concept_false)} samples, input_dim={n_samples_false}")

        # Prepare DataLoader
        is_cuda = device == "cuda"

        def make_loader(dataset, batch_size, is_cuda):
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,  # order doesn't matter
                pin_memory=is_cuda,
                num_workers=32 if is_cuda else 0,
                prefetch_factor=8 if is_cuda else None,
                persistent_workers=is_cuda,
            )

        # Compute sequentially
        print("Computing activations for 'concept=true'...")
        loader_true = make_loader(dataset_concept_true, args.batch_size, is_cuda)
        sum_true = compute_sums(loader_true, sae, device, config["nb_concepts"])

        print("\nComputing activations for 'concept=false'...")
        loader_false = make_loader(dataset_concept_false, args.batch_size, is_cuda)
        sum_false = compute_sums(loader_false, sae, device, config["nb_concepts"])

        # Calculate score
        epsilon = args.epsilon
        normalized_mean_true = sum_true / (sum_true.sum() + epsilon * sum_true.shape[0])
        normalized_mean_false = sum_false / (sum_false.sum() + epsilon * sum_false.shape[0])
        feature_scores = normalized_mean_true - normalized_mean_false
        feature_scores_np = feature_scores.numpy()

        # Save scores
        feature_scores_path = Path(args.feature_scores_path)
        feature_scores_path.parent.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print("SAVING FEATURE SCORES")
        print("=" * 80)
        print(f"Saving feature scores to {feature_scores_path}...")
        torch.save(feature_scores_np, feature_scores_path)
        print("✓ Feature scores saved successfully.")

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
