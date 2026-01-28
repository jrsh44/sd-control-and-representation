#!/usr/bin/env python3
"""
Script to extract text captions from the Conceptual Captions 3M dataset.
Uses the text-only source to avoid binary schema errors.

Supports randomized splitting for both:
1. Native Splits (Preferred): Uses 'train' and 'validation' splits if available.
2. Full dataset processing (Limit=None): Streams data, keeping a random buffer for validation.
3. Subset processing (Limit=N): Reservoir samples N items from the FULL dataset, then splits.

Usage:
    uv run scripts/data/extract_cc3m_prompts.py \
        --train-file path/to/train.txt \
        --val-file path/to/validation.txt \
        --val-size 5000 \
        --limit 50000
"""

import argparse
import random
import sys
from pathlib import Path

from datasets import get_dataset_split_names, load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Extract text from CC3M")
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/cc3m-wds/train.txt",
        help="Path to save the training prompts file",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/cc3m-wds/validation.txt",
        help="Path to save the validation prompts file",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=10000,
        help="Number of prompts to use for validation (default: 10000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total number of prompts to sample. If set, performs reservoir sampling over FULL dataset.",  # noqa: E501
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Buffer size for writing to file",
    )
    args = parser.parse_args()

    # Safety check: ensure val_size is an integer
    val_limit = args.val_size if args.val_size is not None else 10000

    # Ensure directories exist
    train_path = Path(args.train_file)
    val_path = Path(args.val_file)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = "google-research-datasets/conceptual_captions"
    print(f"🌊 Checking dataset '{dataset_name}'...")

    # Check for native splits
    use_native_splits = False
    try:
        splits = get_dataset_split_names(dataset_name)
        if "train" in splits and "validation" in splits:
            use_native_splits = True
            print("   ✅ Detected native 'train' and 'validation' splits. Using them.")
        else:
            print("   ℹ️  No native validation split found. Will perform manual splitting.")
    except Exception as e:
        print(f"   ⚠️  Could not check splits ({e}). Defaulting to manual splitting.")

    # Open files handles
    f_train = open(train_path, "w", encoding="utf-8")
    f_val = open(val_path, "w", encoding="utf-8")

    global_id = 1
    total_processed = 0
    kept_count = 0

    try:
        # --- PATH A: NATIVE SPLITS ---
        if use_native_splits:
            # 1. Process Validation Split
            print(f"\nProcessing Validation Split (Target: {val_limit})...")
            val_dataset = load_dataset(dataset_name, split="validation", streaming=True)

            val_count = 0
            # If limit is set, validation shouldn't exceed the total limit
            actual_val_limit = val_limit
            if args.limit is not None:
                actual_val_limit = min(val_limit, args.limit)

            for sample in tqdm(val_dataset):
                if val_count >= actual_val_limit:
                    break

                prompt = sample.get("caption")
                if prompt:
                    clean = prompt.strip().replace("\n", " ").replace("\r", "")
                    f_val.write(f"{global_id};{clean}\n")
                    global_id += 1
                    val_count += 1
                    total_processed += 1

            print(f"   Extracted {val_count} validation prompts.")

            # 2. Process Train Split
            # If limit is set, we take (limit - val_extracted) samples
            train_goal = float("inf")
            if args.limit is not None:
                train_goal = max(0, args.limit - val_count)

            if train_goal > 0:
                print(
                    f"\nProcessing Train Split (Target: {train_goal if args.limit else 'ALL'})..."
                )
                train_dataset = load_dataset(dataset_name, split="train", streaming=True)

                # Shuffle training data to get random samples if limit is set
                train_dataset = train_dataset.shuffle(seed=42, buffer_size=10000)

                train_count = 0
                for sample in tqdm(
                    train_dataset, total=3_318_333 if args.limit is None else train_goal
                ):
                    if args.limit is not None and train_count >= train_goal:
                        break

                    prompt = sample.get("caption")
                    if prompt:
                        clean = prompt.strip().replace("\n", " ").replace("\r", "")
                        f_train.write(f"{global_id};{clean}\n")
                        global_id += 1
                        train_count += 1
                        total_processed += 1

                print(f"   Extracted {train_count} training prompts.")

            kept_count = total_processed

        # --- PATH B: MANUAL SPLIT (Reservoir Sampling) ---
        else:
            if args.limit:
                print("   Mode: Subset Sampling (Reservoir)")
                print(f"   Goal: Sample {args.limit} random prompts from full dataset")
                print(f"   Split: {val_limit} Val / {max(0, args.limit - val_limit)} Train")
            else:
                print("   Mode: Full Processing")
                print("   Goal: Process all available prompts")
                print(f"   Split: {val_limit} random Val / Rest Train")

            dataset = load_dataset(dataset_name, split="train", streaming=True)
            reservoir = []
            processed_count = 0  # Items seen in stream

            # Total size of CC3M train split is roughly 3,318,333
            pbar = tqdm(dataset, total=3_318_333)

            for sample in pbar:
                prompt = sample.get("caption")
                if not prompt:
                    continue

                clean_prompt = prompt.strip().replace("\n", " ").replace("\r", "")

                # Logic: Subset Sampling (Limit set)
                if args.limit is not None:
                    if len(reservoir) < args.limit:
                        reservoir.append(clean_prompt)
                    else:
                        j = random.randint(0, processed_count)  # noqa: S311
                        if j < args.limit:
                            reservoir[j] = clean_prompt

                # Logic: Full Dataset (Limit None)
                else:
                    if len(reservoir) < val_limit:
                        reservoir.append(clean_prompt)
                    else:
                        j = random.randint(0, processed_count)  # noqa: S311
                        if j < val_limit:
                            evicted_prompt = reservoir[j]
                            reservoir[j] = clean_prompt
                            f_train.write(f"{global_id};{evicted_prompt}\n")
                            global_id += 1
                        else:
                            f_train.write(f"{global_id};{clean_prompt}\n")
                            global_id += 1

                processed_count += 1

            # Finalize Reservoir
            if args.limit is not None:
                print(f"\nSampling complete. Finalizing {len(reservoir)} items...")
                random.shuffle(reservoir)
                val_items = reservoir[:val_limit]
                train_items = reservoir[val_limit:]

                for p in val_items:
                    f_val.write(f"{global_id};{p}\n")
                    global_id += 1
                for p in train_items:
                    f_train.write(f"{global_id};{p}\n")
                    global_id += 1
            else:
                print(f"\nStreaming complete. Writing {len(reservoir)} validation prompts...")
                for p in reservoir:
                    f_val.write(f"{global_id};{p}\n")
                    global_id += 1

            kept_count = global_id - 1

        print("\n✅ Extraction complete!")
        print(f"   Total saved:   {kept_count}")
        print(f"   Validation:    {val_path}")
        print(f"   Train:         {train_path}")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted. Closing files...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        f_train.close()
        f_val.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
