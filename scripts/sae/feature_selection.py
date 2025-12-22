#!/usr/bin/env python3
"""
Example usage:
    python scripts/sae/feature_selection.py \
        --dataset_path /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/nudity/representations/unet_up_1_att_1 \
        --dataset_name nudity \
        --concept object \
        --concept_value exposed anus \
        --sae_dir_path /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp8_topk16_lr4em4_ep5_bs4096 \
        --features_dir_path /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp8_topk16_lr4em4_ep5_bs4096/feature_sums \
        --feature_sums_prefix unet_up_1_att_1_concept_object_exposed_anus \
        --epsilon 1e-8

"""  # noqa: E501

import argparse
import hashlib
import os
import platform
import re
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
        "--sae_dir_path",
        type=str,
        required=True,
        help="Path to directory where SAE model is stored, in this folder there is single .pt file",
    )
    # feature sums directory and prefix
    parser.add_argument(
        "--features_dir_path",
        type=str,
        required=True,
        help="Directory where partial feature sum files will be saved",
    )
    parser.add_argument(
        "--feature_sums_prefix",
        type=str,
        required=True,
        help="Prefix for partial feature sum files (e.g., 'unet_mid_att_concept_object_cat')",
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
    print(f"SAE path: {args.sae_dir_path}")
    print(f"Features dir: {args.features_dir_path}")
    print(f"Feature sums prefix: {args.feature_sums_prefix}")
    print(f"Device: {device}")
    print("-" * 80)

    # --------------------------------------------------------------------------
    # 2. Initialize WandB
    # --------------------------------------------------------------------------
    if not args.skip_wandb:
        wandb.login()

        # Create a run name
        sae_stem = Path(args.sae_dir_path).stem
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
                "sae_dir_path": args.sae_dir_path,
                "architecture": "TopKSAE",
                # "top_k": args.top_k,
            },
            "feature_selection": {
                # "epsilon": args.epsilon,
                # "batch_size": args.batch_size,
                "log_every": args.log_every,
                "features_dir_path": args.features_dir_path,
                "feature_sums_prefix": args.feature_sums_prefix,
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
        sae_dir_path = Path(args.sae_dir_path)
        if not sae_dir_path.exists():
            raise FileNotFoundError(f"SAE directory not found: {sae_dir_path}")

        # Find the first model.pt or checkpoint.pt file in the sae_dir_path
        pt_files = list(sae_dir_path.glob("model.pt")) + list(sae_dir_path.glob("checkpoint.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No model.pt or checkpoint.pt file found in {sae_dir_path}")

        sae_path = pt_files[0]
        print(f"Using SAE weights file: {sae_path}")

        # extract topk and batch_size from sae_path e.g. wds_nudity/unet_up_1_att_1/exp8_topk16_lr4em4_ep5_bs4096  # noqa: E501
        match = re.search(r"topk(\d+)_.*_bs(\d+)", str(sae_path))
        if match:
            extracted_topk = int(match.group(1))
            extracted_batch_size = int(match.group(2))
            print(
                f"Extracted top_k={extracted_topk}, batch_size={extracted_batch_size} from SAE path"
            )
        else:
            print("Warning: Could not extract top_k and batch_size from SAE path")
            extracted_topk = 32
            extracted_batch_size = 4096
            print(f"Using default top_k={extracted_topk}, batch_size={extracted_batch_size}")

        # Load state dict
        state_dict = torch.load(sae_path, map_location="cpu")

        # print keys in state_dict (for tests)
        print(f"SAE state_dict keys: {list(state_dict.keys())}")

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
            top_k=extracted_topk,
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
            """Create optimized DataLoader for big data processing."""
            import os

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

            # Optimize num_workers based on available CPUs
            if is_cuda:
                slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
                if slurm_cpus:
                    available_cpus = int(slurm_cpus)
                else:
                    available_cpus = os.cpu_count() or 4

                # Use most CPUs but leave 1-2 for main process
                num_workers = max(4, min(available_cpus - 2, 12))  # Cap at 12 to avoid overhead
                prefetch_factor = 4  # Increased prefetch for better pipeline
                print(f"DataLoader: {num_workers} workers, prefetch={prefetch_factor}")
            else:
                num_workers = 0
                prefetch_factor = None

            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=is_cuda,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=num_workers > 0,
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
        loader_false = make_loader(dataset_concept_false, extracted_batch_size, is_cuda)
        sums_false, counts_false = compute_sums_per_timestep(
            loader_false, sae, device, nb_concepts, args.log_every, "compute_sums_false"
        )
        print("✓ Sums for 'concept=false' computed")

        # Clean up loader and force garbage collection
        del loader_false
        if is_cuda:
            torch.cuda.empty_cache()
        print("  Memory cleaned after false dataset processing")

        print("\nComputing activations for 'concept=true'...")
        loader_true = make_loader(dataset_concept_true, extracted_batch_size, is_cuda)
        sums_true, counts_true = compute_sums_per_timestep(
            loader_true, sae, device, nb_concepts, args.log_every, "compute_sums_true"
        )
        print("✓ Sums for 'concept=true' computed")

        # Clean up loader
        del loader_true
        if is_cuda:
            torch.cuda.empty_cache()
        print("  Memory cleaned after true dataset processing")

        # --------------------------------------------------------------------------
        # 6. Save Results
        # --------------------------------------------------------------------------
        features_dir = Path(args.features_dir_path)
        features_dir.mkdir(parents=True, exist_ok=True)

        feature_sums = {
            "sums_true_per_timestep": sums_true,  # dict[int, Tensor]
            "counts_true_per_timestep": counts_true,  # dict[int, int]
            "sums_false_per_timestep": sums_false,  # dict[int, Tensor]
            "counts_false_per_timestep": counts_false,  # dict[int, int]
            "timesteps": sorted(set(sums_true.keys()) | set(sums_false.keys())),
        }

        job_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        dataset_hash = hashlib.md5(args.dataset_path.encode()).hexdigest()[:8]  # noqa: S324
        partial_file = f"{args.feature_sums_prefix}_job{job_id}_{dataset_hash}.pt"
        partial_path = features_dir / partial_file

        print("\n" + "=" * 80)
        print("SAVING FEATURE SUMS")
        print("=" * 80)
        print(f"Saving feature sums to {partial_path}...")
        torch.save(feature_sums, partial_path)
        print("✓ Feature sums saved successfully.")

        if not args.skip_wandb:
            wandb.log(
                {
                    "final/samples_concept_true": n_samples_true,
                    "final/samples_concept_false": n_samples_false,
                    "final/timesteps_processed": len(feature_sums["timesteps"]),
                    "final/feature_sums_path": str(partial_path),
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
