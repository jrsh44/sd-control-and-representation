#!/usr/bin/env python3
"""
Alternative test script for Eden cluster - tests different configurations.
This script uses different parameters and output format.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Alternative test script for cluster")
    parser.add_argument("--backbone", type=str, required=True, help="Model backbone name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--workers", type=int, default=12, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-split", action="store_true", help="Use training split")

    args = parser.parse_args()

    # Get SLURM environment variables
    job_id = os.environ.get("SLURM_JOB_ID", "N/A")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "N/A")

    print("=" * 50)
    print("EDEN TEST - Script 2")
    print("=" * 50)
    print(f"[Job {job_id} | Task {task_id}]")
    print(f"[Backbone: {args.backbone}]")
    print(f"[Dataset: {args.dataset}]")
    print(f"[Batch: {args.batch_size} | Workers: {args.workers} | Seed: {args.seed}]")
    print(f"[Split: {'train' if args.train_split else 'validation'}]")
    print("=" * 50)

    # Write simple result file
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "alternative")
    os.makedirs(results_dir, exist_ok=True)

    result_file = os.path.join(results_dir, f"alternative_test_task_{task_id}.txt")
    with open(result_file, "w") as f:
        f.write(f"Task {task_id}\n")
        f.write(f"Job: {job_id}\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Workers: {args.workers}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Split: {'train' if args.train_split else 'validation'}\n")

    print(f"Results: {result_file}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        sys.exit(1)
