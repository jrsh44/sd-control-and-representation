#!/usr/bin/env python3
"""
Example usage:
    python scripts/sae/feature_selection.py \
        --dataset_path /mnt/evafs/groups/mi2lab/mjarosz/results_npy/finetuned_sd_saeuron/cached_representations/unet_up_1_att_1 \
        --concept object \
        --concept_value cats \
        --sae_path /mnt/evafs/groups/mi2lab/mjarosz/results_npy/finetuned_sd_saeuron/sae/unet_up_1_att_1_sae.pt \
        --feature_scores_path /mnt/evafs/groups/mi2lab/mjarosz/results_npy/finetuned_sd_saeuron/sae_scores/unet_up_1_att_1_concept_object_cat.npy \
        --epsilon 1e-8 \
        --batch_size 4096 \
        --top_k 32

"""  # noqa: E501

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
    compute_means,
    concept_filtering_function,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature selection using pretrained Sparse Autoencoder (SAE) given a concept",
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
    # feature means path
    parser.add_argument(
        "--feature_means_path",
        type=str,
        required=True,
        help="Path to .npy file for feature means (load if exists, create and save if not)",
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
    # top_k
    parser.add_argument(
        "--top_k",
        type=int,
        required=True,
        help="Top-k sparsity for SAE",
    )
    # log every
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        required=False,
        help="Log every N batches",
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
    print(f"Feature means path: {args.feature_means_path}")
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
                name=f"feature_selection_SAE_{Path(args.sae_path).stem}",
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

        # Load state dict
        state_dict = torch.load(sae_path, map_location="cpu")

        # Automatyczne wykrycie i usunięcie prefiksu _orig_mod.
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            print("Detected torch.compile() prefix → removing '_orig_mod.'")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Wnioskowanie wymiarów
        enc_weight = state_dict["encoder.final_block.0.weight"]
        input_shape = enc_weight.shape[1]
        nb_concepts = enc_weight.shape[0]
        print(f"Inferred input_dim={input_shape}, nb_concepts={nb_concepts}")

        # Tworzymy model i ładujemy wagi
        sae = TopKSAE(
            input_shape=input_shape,
            nb_concepts=nb_concepts,
            top_k=args.top_k,
            device=device,
        )
        sae.load_state_dict(state_dict)  # teraz działa idealnie
        sae = sae.to(device)
        sae.eval()
        print("SAE loaded and moved to device")

        # Prepare dataloader functions
        is_cuda = device == "cuda"

        def make_loader(dataset, batch_size, is_cuda):
            if torch.cuda.is_available() and hasattr(torch._dynamo.external_utils, "is_compiled"):
                # Jeśli model jest skompilowany – wyłącz wieloprocesowość (unikamy deadlocka)
                print("torch.compile() detected → using num_workers=0 to avoid deadlock")
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=0,  # ← KLUCZOWA ZMIANA
                )
            else:
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=is_cuda,
                    num_workers=6 if is_cuda else 0,
                    prefetch_factor=2 if is_cuda else None,
                    persistent_workers=is_cuda,
                )

        # Load dataset
        dataset_path = Path(args.dataset_path)
        cache_dir = dataset_path.parent
        layer_name = dataset_path.name

        # === CONCEPT FALSE ===
        dataset_concept_false = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            filter_fn=concept_filtering_function(args.concept, args.concept_value, negate=True),
            return_metadata=False,
        )
        n_samples_false = dataset_concept_false._full_data.shape[0]
        print(f"Loaded dataset: {len(dataset_concept_false)} samples, input_dim={n_samples_false}")

        print("\nComputing activations for 'concept=false'...")
        loader_false = make_loader(dataset_concept_false, args.batch_size, is_cuda)
        mean_false = compute_means(
            loader_false, sae, device, nb_concepts, args.log_every, "compute_means_false"
        )  # noqa: E501
        print("Means for 'concept=false' computed")

        # === CONCEPT TRUE ===
        dataset_concept_true = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            filter_fn=concept_filtering_function(args.concept, args.concept_value),
            return_metadata=False,
        )
        n_samples_true = dataset_concept_true._full_data.shape[0]
        print(f"Loaded dataset: {len(dataset_concept_true)} samples, input_dim={n_samples_true}")

        # Compute sequentially
        print("Computing activations for 'concept=true'...")
        loader_true = make_loader(dataset_concept_true, args.batch_size, is_cuda)
        mean_true = compute_means(
            loader_true, sae, device, nb_concepts, args.log_every, "compute_means_true"
        )  # noqa: E501
        print("Means for 'concept=true' computed")

        # Save means
        feature_means_path = Path(args.feature_means_path)
        feature_means_path.parent.mkdir(parents=True, exist_ok=True)
        feature_means = {
            "mean_true": mean_true,
            "mean_false": mean_false,
        }
        print("\n" + "=" * 80)
        print("SAVING FEATURE MEANS")
        print("=" * 80)
        print(f"Saving feature means to {feature_means_path}...")
        torch.save(feature_means, feature_means_path)
        print("✓ Feature means saved successfully.")

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
