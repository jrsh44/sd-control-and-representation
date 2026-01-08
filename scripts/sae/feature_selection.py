#!/usr/bin/env python3
"""
Example usage:
    python scripts/sae/feature_selection.py \
        --dataset_path /mnt/evafs/groups/mi2lab/mjarosz/results_npy/finetuned_sd_saeuron/unlearn_canvas/representations/train/unet_up_1_att_1 \
        --concept object \
        --concept_value cats \
        --sae_path /mnt/evafs/groups/mi2lab/mjarosz/results_npy/finetuned_sd_saeuron/sae/unet_up_1_att_1_sae.pt \
        --feature_sums_path /mnt/evafs/groups/mi2lab/mjarosz/results_npy/finetuned_sd_saeuron/sae_scores/unet_up_1_att_1_concept_object_cat.npy \
        --epsilon 1e-8 \
        --batch_size 4096 \
        --top_k 32

"""  # noqa: E501

import argparse
import platform
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
    compute_sums_per_timestep,
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
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="unlearn_canvas",
        help="Name of the dataset (used for WandB logging and organization)",
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
        "--feature_sums_path",
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
    # --------------------------------------------------------------------------
    # 1. Parse Arguments and Setup
    # --------------------------------------------------------------------------
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device == "cuda"

    print("=" * 80)
    print("FEATURE SELECTION USING PRETRAINED SAE")
    print("=" * 80)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Concept: {args.concept}")
    print(f"Concept value: {args.concept_value}")
    print(f"SAE path: {args.sae_path}")
    print(f"Feature sums path: {args.feature_sums_path}")
    print(f"Device: {device}")
    print("-" * 80)

    # --------------------------------------------------------------------------
    # 2. Initialize WandB
    # --------------------------------------------------------------------------
    if not args.skip_wandb:
        wandb.login()

        # Create a run name
        sae_stem = Path(args.sae_path).stem
        run_name = (
            f"FeatureSelection_{args.dataset_name}_{sae_stem}_{args.concept}_{args.concept_value}"
        )

        gpu_name = torch.cuda.get_device_name(0) if is_cuda else "Unknown"

        # Structured Configuration
        config = {
            "dataset": {
                "name": args.dataset_name,
                "path": args.dataset_path,
                "layer_name": Path(args.dataset_path).name,
            },
            "concept": {
                "name": args.concept,
                "value": args.concept_value,
            },
            "model": {
                "sae_path": args.sae_path,
                "architecture": "TopKSAE",
                "top_k": args.top_k,
            },
            "feature_selection": {
                "epsilon": args.epsilon,
                "batch_size": args.batch_size,
                "log_every": args.log_every,
                "feature_sums_path": args.feature_sums_path,
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
            group=f"feature_selection_{args.dataset_name}_{args.concept}",
            job_type="feature_selection",
            tags=["sae", "feature_selection", args.dataset_name, args.concept, args.concept_value],
            notes="Feature selection using pretrained SAE for concept analysis.",
        )
        print(f"🚀 WandB Initialized: {run_name}")

    # --------------------------------------------------------------------------
    # 3. Load SAE Model
    # --------------------------------------------------------------------------
    try:
        sae_path = Path(args.sae_path)
        if not sae_path.exists():
            raise FileNotFoundError(f"SAE not found: {sae_path}")

        # Load state dict
        state_dict = torch.load(sae_path, map_location="cpu")

        # Auto-detect and remove _orig_mod prefix from torch.compile()
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            print("Detected torch.compile() prefix → removing '_orig_mod.'")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Infer dimensions from weights
        enc_weight = state_dict["encoder.final_block.0.weight"]
        input_shape = enc_weight.shape[1]
        nb_concepts = enc_weight.shape[0]
        print(f"Inferred input_dim={input_shape}, nb_concepts={nb_concepts}")

        # Create model and load weights
        sae = TopKSAE(
            input_shape=input_shape,
            nb_concepts=nb_concepts,
            top_k=args.top_k,
            device=device,
        )
        sae.load_state_dict(state_dict)
        sae = sae.to(device)
        sae.eval()
        print("✓ SAE loaded and moved to device")

        # --------------------------------------------------------------------------
        # 4. Setup Datasets and DataLoaders
        # --------------------------------------------------------------------------
        def make_loader(dataset, batch_size, is_cuda):
            if torch.cuda.is_available() and hasattr(torch._dynamo.external_utils, "is_compiled"):
                # If model is compiled – disable multiprocessing (avoid deadlock)
                print("torch.compile() detected → using num_workers=0 to avoid deadlock")
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=0,
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

        # Load dataset paths
        dataset_path = Path(args.dataset_path)
        cache_dir = dataset_path.parent
        layer_name = dataset_path.name

        # === CONCEPT FALSE ===
        dataset_concept_false = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            filter_fn=concept_filtering_function(args.concept, args.concept_value, negate=True),
            return_metadata=False,
            return_timestep=True,
        )
        n_samples_false = dataset_concept_false._full_data.shape[0]
        print(f"Loaded 'concept=false' dataset: {len(dataset_concept_false)} samples")

        # === CONCEPT TRUE ===
        dataset_concept_true = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            filter_fn=concept_filtering_function(args.concept, args.concept_value),
            return_metadata=False,
            return_timestep=True,
        )
        n_samples_true = dataset_concept_true._full_data.shape[0]
        print(f"Loaded 'concept=true' dataset: {len(dataset_concept_true)} samples")

        # --------------------------------------------------------------------------
        # 5. Compute Feature Sums
        # --------------------------------------------------------------------------
        print("\nComputing activations for 'concept=false'...")
        loader_false = make_loader(dataset_concept_false, args.batch_size, is_cuda)
        sums_false, counts_false = compute_sums_per_timestep(
            loader_false, sae, device, nb_concepts, args.log_every, "compute_sums_false"
        )
        print("✓ Sums for 'concept=false' computed")

        print("\nComputing activations for 'concept=true'...")
        loader_true = make_loader(dataset_concept_true, args.batch_size, is_cuda)
        sums_true, counts_true = compute_sums_per_timestep(
            loader_true, sae, device, nb_concepts, args.log_every, "compute_sums_true"
        )
        print("✓ Sums for 'concept=true' computed")

        # --------------------------------------------------------------------------
        # 6. Save Results
        # --------------------------------------------------------------------------
        feature_sums_path = Path(args.feature_sums_path)
        feature_sums_path.parent.mkdir(parents=True, exist_ok=True)
        feature_sums = {
            "sums_true_per_timestep": sums_true,  # dict[int, Tensor]
            "counts_true_per_timestep": counts_true,  # dict[int, int]
            "sums_false_per_timestep": sums_false,  # dict[int, Tensor]
            "counts_false_per_timestep": counts_false,  # dict[int, int]
            "timesteps": sorted(set(sums_true.keys()) | set(sums_false.keys())),
        }

        print("\n" + "=" * 80)
        print("SAVING FEATURE SUMS")
        print("=" * 80)
        print(f"Saving feature sums to {feature_sums_path}...")
        torch.save(feature_sums, feature_sums_path)
        print("✓ Feature sums saved successfully.")

        if not args.skip_wandb:
            wandb.log(
                {
                    "final/samples_concept_true": n_samples_true,
                    "final/samples_concept_false": n_samples_false,
                    "final/timesteps_processed": len(feature_sums["timesteps"]),
                    "final/feature_sums_path": str(feature_sums_path),
                }
            )

        print("\n" + "=" * 80)
        print("Done.")
        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    finally:
        if not args.skip_wandb:
            wandb.finish()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
