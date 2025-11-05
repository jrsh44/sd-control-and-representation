#!/usr/bin/env python3
"""
ONLY TEST FILE
"""

"""
EXAMPLE USAGE:
uv run src/read_cached_rep.py ../results/sd_1_5/TEXT_EMBEDDING_FINAL/cached_rep.pt --detailed
"""

import argparse
import sys
from pathlib import Path

import torch


def print_cache_format(cache_path: Path, detailed: bool = False):
    """
    Print the structure and format of a cached representation file.
    
    Args:
        cache_path: Path to the cached_rep.pt file
        detailed: If True, print detailed information including tensor shapes
    """
    if not cache_path.exists():
        print(f"ERROR: Cache file does not exist: {cache_path}")
        return 1
    
    try:
        cache = torch.load(cache_path, map_location='cpu')
    except Exception as e:
        print(f"ERROR: Could not load cache from {cache_path}: {e}")
        return 1
    
    print("=" * 70)
    print(f"CACHE FORMAT: {cache_path.name}")
    print("=" * 70)
    print(f"File: {cache_path}")
    print(f"Type: {type(cache).__name__}")
    
    if not isinstance(cache, dict):
        print(f"\nUnexpected cache type: {type(cache)}")
        return 0
    
    # Count totals
    num_objects = len(cache)
    num_styles = sum(len(styles) for styles in cache.values())
    num_prompts = sum(
        len(prompts)
        for obj_cache in cache.values()
        for prompts in obj_cache.values()
    )
    
    print(f"\nSummary:")
    print(f"  Objects: {num_objects}")
    print(f"  Object-Style combinations: {num_styles}")
    print(f"  Total cached prompts: {num_prompts}")
    
    print("\nStructure: {object: {style: {prompt_nr: tensor}}}")
    print("\nObjects:")
    
    for obj_name in sorted(cache.keys()):
        obj_cache = cache[obj_name]
        print(f"\n  [{obj_name}]")
        
        for style_name in sorted(obj_cache.keys()):
            style_cache = obj_cache[style_name]
            prompt_nrs = sorted(style_cache.keys())
            
            if detailed:
                print(f"    {style_name}: {len(prompt_nrs)} prompts")
                
                # Show a sample tensor shape
                if prompt_nrs:
                    sample_nr = prompt_nrs[0]
                    sample_tensor = style_cache[sample_nr]
                    print(f"      Sample (prompt {sample_nr}): shape={sample_tensor.shape}, dtype={sample_tensor.dtype}")
                    
                    # Show all prompt numbers
                    print(f"      Prompt numbers: {prompt_nrs}")
            else:
                print(f"    {style_name}: {len(prompt_nrs)} prompts (numbers: {prompt_nrs})")
    
    print("\n" + "=" * 70)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Print the format and structure of cached representations"
    )
    parser.add_argument(
        "cache_path",
        type=str,
        help="Path to the cached_rep.pt file (e.g., results/sd_1_5/TEXT_EMBEDDING_FINAL/cached_rep.pt)",
    )
    parser.add_argument(
        "-d", "--detailed",
        action="store_true",
        help="Print detailed information including tensor shapes",
    )
    
    args = parser.parse_args()
    cache_path = Path(args.cache_path)
    
    return print_cache_format(cache_path, detailed=args.detailed)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        sys.exit(1)
