#!/usr/bin/env python3
"""
Merge temp metadata files and cleanup after interrupted cache generation.

Use this script when a cache generation job was interrupted (e.g., SLURM time limit)
and left behind temp files like:
  - metadata_process_*.jsonl
  - counter.txt
  - *.lock

Usage:
  # Merge a specific layer:
  uv run scripts/sd_v1_5/merge_cache_metadata.py \
    --cache-dir /path/to/representations \
    --layer unet_up_1_att_1

  # Merge all layers in a cache directory:
  uv run scripts/sd_v1_5/merge_cache_metadata.py \
    --cache-dir /path/to/representations \
    --all

  # Dry run (show what would be done):
  uv run scripts/sd_v1_5/merge_cache_metadata.py \
    --cache-dir /path/to/representations \
    --all --dry-run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")


def get_layer_status(layer_dir: Path) -> Dict:
    """Get status of a layer directory."""
    status = {
        "name": layer_dir.name,
        "path": str(layer_dir),
        "has_data": (layer_dir / "data.npy").exists(),
        "has_metadata": (layer_dir / "metadata.json").exists(),
        "temp_files": list(layer_dir.glob("metadata_process_*.jsonl")),
        "lock_files": list(layer_dir.glob("*.lock")),
        "has_counter": (layer_dir / "counter.txt").exists(),
    }

    # Count entries in metadata.json
    if status["has_metadata"]:
        try:
            with open(layer_dir / "metadata.json", "r") as f:
                info = json.load(f)
            status["metadata_entries"] = len(info.get("entries", []))
            status["is_finalized"] = info.get("finalized", False)
        except Exception:
            status["metadata_entries"] = 0
            status["is_finalized"] = False
    else:
        status["metadata_entries"] = 0
        status["is_finalized"] = False

    temp_entries = 0
    for temp_file in status["temp_files"]:
        with open(temp_file, "r") as f:
            temp_entries += sum(1 for line in f if line.strip())
    status["temp_entries"] = temp_entries

    status["needs_merge"] = len(status["temp_files"]) > 0
    status["needs_cleanup"] = len(status["lock_files"]) > 0 or status["has_counter"]

    return status


def merge_layer(layer_dir: Path, dry_run: bool = False) -> Dict:
    """
    Merge temp metadata files into metadata.json and cleanup.

    Args:
        layer_dir: Path to layer directory
        dry_run: If True, only show what would be done

    Returns:
        dict with merge statistics
    """
    status = get_layer_status(layer_dir)
    result = {
        "layer": layer_dir.name,
        "success": False,
        "entries_merged": 0,
        "files_cleaned": 0,
        "errors": [],
    }

    if not status["has_data"]:
        result["errors"].append("No data.npy found")
        return result

    metadata_path = layer_dir / "metadata.json"

    # Load existing metadata
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                info = json.load(f)
        except Exception as e:
            result["errors"].append(f"Failed to read metadata.json: {e}")
            return result
    else:
        info = {"entries": []}

    existing_entries = info.get("entries", [])
    # Use (prompt_nr, object) as unique key to handle class-based generation
    existing_keys = {(e["prompt_nr"], e.get("object", "")) for e in existing_entries}

    # Read temp files
    new_entries = []
    for temp_file in sorted(status["temp_files"]):
        try:
            with open(temp_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        key = (entry["prompt_nr"], entry.get("object", ""))
                        if key not in existing_keys:
                            new_entries.append(entry)
                            existing_keys.add(key)
        except Exception as e:
            result["errors"].append(f"Error reading {temp_file.name}: {e}")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Layer: {layer_dir.name}")
    print(f"  Existing entries: {len(existing_entries)}")
    print(f"  New entries from temp files: {len(new_entries)}")
    print(f"  Temp files: {len(status['temp_files'])}")
    print(f"  Lock files: {len(status['lock_files'])}")
    print(f"  Counter file: {'Yes' if status['has_counter'] else 'No'}")

    if dry_run:
        result["entries_merged"] = len(new_entries)
        result["files_cleaned"] = (
            len(status["temp_files"])
            + len(status["lock_files"])
            + (1 if status["has_counter"] else 0)
        )
        result["success"] = True
        return result

    # Merge entries
    if new_entries:
        all_entries = existing_entries + new_entries
        info["entries"] = all_entries
        info["entry_count"] = len(all_entries)
        info["finalized"] = True
        info["merge_timestamp"] = time.time()
        info["merge_script"] = "merge_cache_metadata.py"

        try:
            with open(metadata_path, "w") as f:
                json.dump(info, f, indent=2)
            result["entries_merged"] = len(new_entries)
            print(f"  ✓ Merged {len(new_entries)} entries into metadata.json")
        except Exception as e:
            result["errors"].append(f"Failed to write metadata.json: {e}")
            return result
    else:
        # Just mark as finalized if no new entries
        if not info.get("finalized"):
            info["finalized"] = True
            info["merge_timestamp"] = time.time()
            try:
                with open(metadata_path, "w") as f:
                    json.dump(info, f, indent=2)
            except Exception as e:
                result["errors"].append(f"Failed to update metadata.json: {e}")
        print("  ✓ No new entries to merge")

    # Cleanup temp files
    for temp_file in status["temp_files"]:
        try:
            temp_file.unlink()
            result["files_cleaned"] += 1
        except Exception as e:
            result["errors"].append(f"Failed to delete {temp_file.name}: {e}")

    # Cleanup lock files
    for lock_file in status["lock_files"]:
        try:
            lock_file.unlink()
            result["files_cleaned"] += 1
        except Exception as e:
            result["errors"].append(f"Failed to delete {lock_file.name}: {e}")

    # Cleanup counter
    counter_file = layer_dir / "counter.txt"
    if counter_file.exists():
        try:
            counter_file.unlink()
            result["files_cleaned"] += 1
        except Exception as e:
            result["errors"].append(f"Failed to delete counter.txt: {e}")

    print(f"  ✓ Cleaned up {result['files_cleaned']} files")

    result["success"] = len(result["errors"]) == 0
    return result


def find_layers(cache_dir: Path) -> List[Path]:
    """Find all layer directories in cache."""
    layers = []
    for subdir in cache_dir.iterdir():
        if subdir.is_dir() and (subdir / "data.npy").exists():
            layers.append(subdir)
    return sorted(layers)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Merge temp metadata files and cleanup after interrupted cache generation"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Path to cache directory (default: $RESULTS_DIR from .env)",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Specific layer name to merge (e.g., unet_up_1_att_1)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Merge all layers in the cache directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Only show status of layers, don't merge",
    )
    return parser.parse_args()


def main():
    # --------------------------------------------------------------------------
    # 1. Parse Arguments and Setup
    # --------------------------------------------------------------------------
    args = parse_args()

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    elif os.environ.get("RESULTS_DIR"):
        cache_dir = Path(os.environ["RESULTS_DIR"])
        print(f"Using RESULTS_DIR from environment: {cache_dir}")
    else:
        print("ERROR: Cache directory not specified.")
        print("Please either:")
        print("  1. Use --cache-dir argument, or")
        print("  2. Set RESULTS_DIR in .env file")
        return 1

    if not cache_dir.exists():
        print(f"ERROR: Cache directory not found: {cache_dir}")
        return 1

    # Find layers to process
    if args.layer:
        layer_dir = cache_dir / args.layer.lower()
        if not layer_dir.exists():
            print(f"ERROR: Layer directory not found: {layer_dir}")
            return 1
        layers = [layer_dir]
    elif args.all:
        layers = find_layers(cache_dir)
        if not layers:
            print(f"No layer directories found in {cache_dir}")
            return 1
    else:
        print("ERROR: Specify --layer <name> or --all")
        return 1

    # --------------------------------------------------------------------------
    # 2. Display Configuration
    # --------------------------------------------------------------------------
    print("=" * 80)
    print("CACHE METADATA MERGE TOOL")
    print("=" * 80)
    print(f"Cache directory: {cache_dir}")
    print(f"Layers found: {len(layers)}")
    print("=" * 80)

    if args.status:
        print("\nLayer Status:")
        print("-" * 80)
        for layer_dir in layers:
            status = get_layer_status(layer_dir)
            print(f"\n{status['name']}:")
            print(f"  Data file: {'✓' if status['has_data'] else '✗'}")
            print(f"  Metadata entries: {status['metadata_entries']}")
            print(f"  Finalized: {'✓' if status['is_finalized'] else '✗'}")
            print(f"  Temp files: {len(status['temp_files'])} ({status['temp_entries']} entries)")
            print(f"  Lock files: {len(status['lock_files'])}")
            print(f"  Counter: {'Yes' if status['has_counter'] else 'No'}")
            if status["needs_merge"] or status["needs_cleanup"]:
                print(
                    f"  ⚠️  Needs {'merge' if status['needs_merge'] else ''}"
                    f"{' and ' if status['needs_merge'] and status['needs_cleanup'] else ''}"
                    f"{'cleanup' if status['needs_cleanup'] else ''}"
                )
        return 0

    # --------------------------------------------------------------------------
    # 3. Merge Layers
    # --------------------------------------------------------------------------
    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]")

    results = []
    for layer_dir in layers:
        result = merge_layer(layer_dir, dry_run=args.dry_run)
        results.append(result)

    # --------------------------------------------------------------------------
    # 4. Summary
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_merged = sum(r["entries_merged"] for r in results)
    total_cleaned = sum(r["files_cleaned"] for r in results)
    total_errors = sum(len(r["errors"]) for r in results)

    print(f"Layers processed: {len(results)}")
    print(f"Entries merged: {total_merged}")
    print(f"Files cleaned: {total_cleaned}")

    if total_errors > 0:
        print(f"\n⚠️  Errors encountered: {total_errors}")
        for result in results:
            if result["errors"]:
                print(f"  {result['layer']}:")
                for error in result["errors"]:
                    print(f"    - {error}")
        return 1

    print("\n✓ All done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
