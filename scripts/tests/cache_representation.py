#!/usr/bin/env python3
"""
Test cache.py with a real representation from .pt file.

Usage:
    # First, create test representation:
    uv run scripts/tests/create_representation.py --prompt "a cat" --steps 10

    # Then, test cache with it:
    uv run scripts/tests/cache_representation.py
"""

import sys
from pathlib import Path

import torch
from datasets import load_from_disk

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import RepresentationCache  # noqa: E402

print("=" * 70)
print("Testing Cache with Real Representation")
print("=" * 70)

# Setup paths
test_file = Path("test_data/real_representation.pt")
cache_dir = Path("test_cache_output")

# Clean up previous test
if cache_dir.exists():
    import shutil

    shutil.rmtree(cache_dir)
    print("🧹 Cleaned up previous test cache")

cache_dir.mkdir(parents=True, exist_ok=True)

# Check if test data exists
if not test_file.exists():
    print("\n❌ ERROR: Test representation file not found!")
    sys.exit(1)

# Initialize cache
print("\n1. Initializing cache...")
cache = RepresentationCache(cache_dir, use_fp16=True)
print(f"   Cache dir: {cache_dir.absolute()}")

# Load test representation
print("\n2. Loading real representation from .pt file...")
representation = torch.load(test_file)
print(f"   Representation shape: {representation.shape}")

# Expect 4D tensor: [timesteps, batch, spatial, features]
if representation.dim() != 4:
    error_msg = (
        f"Expected 4D tensor [timesteps, batch, spatial, features], got: {representation.shape}"
    )
    print(f"   ❌ {error_msg}")
    sys.exit(1)

timesteps, batch, spatial, features = representation.shape
print(f"   Timesteps: {timesteps}, Batch: {batch}, Spatial: {spatial}, Features: {features}")

layer_type = "spatial"
layer_name = "down_blocks.2.attentions.1"
print(f"   Layer: {layer_name}")

# Test 1: Save representation
print(f"\n3. Saving {layer_type} representation...")
cache.save_representation(
    layer_name=layer_name,
    object_name="cat",
    style="Impressionism",
    prompt_nr=1,
    prompt_text="a cat in Impressionism style",
    representation=representation,
    num_steps=representation.shape[0],
    guidance_scale=7.5,
)
print("   ✅ Saved")

# Test 2: Check tmp_shards exist
print("\n4. Checking temporary shards...")
layer_path = cache_dir / layer_name.lower()
tmp_shards = layer_path / ".tmp_shards"
if tmp_shards.exists():
    shard_count = len(list(tmp_shards.glob("shard_*")))
    print(f"   Found {shard_count} shard(s) ✅")
else:
    print("   ❌ No shards found")
    sys.exit(1)

# Test 3: Consolidate metadata
print("\n5. Consolidating metadata...")
cache.save_metadata()

# Test 4: Verify tmp_shards cleaned up
print("\n6. Verifying cleanup...")
if not tmp_shards.exists():
    print("   tmp_shards cleaned up ✅")
else:
    print("   ❌ tmp_shards still exist")

# Test 5: Check existence
print("\n7. Testing existence check...")
exists = cache.check_exists(layer_name, "cat", "Impressionism", 1)
print(f"   cat/Impressionism/1 exists: {exists} {'✅' if exists else '❌'}")

# Test 6: Load with HuggingFace datasets
print("\n8. Loading with HuggingFace datasets...")
ds = load_from_disk(str(layer_path))
print(f"   Dataset: {len(ds)} records")

# Calculate expected records
timesteps, batch, spatial, features = representation.shape
expected_records = timesteps * spatial
print(f"   Expected: {expected_records} records ({timesteps} timesteps × {spatial} spatial)")

if len(ds) == expected_records:
    print("   ✅ Record count matches")
else:
    print(f"   ❌ Mismatch: expected {expected_records}, got {len(ds)}")

# Test 7: Check dataset structure
print("\n9. Checking dataset structure...")
print(f"   Columns: {ds.column_names}")

first_record = ds[0]
print("\n   First record:")
print(f"     - timestep: {first_record['timestep']}")
print(f"     - spatial: {first_record['spatial']}")
print(f"     - object: {first_record['object']}")
print(f"     - style: {first_record['style']}")
print(f"     - features length: {len(first_record['list_of_features'])}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

success = exists and len(ds) == expected_records

if success:
    print("✅ All tests passed!")
else:
    print("❌ Some tests failed!")

print(f"\nCache location: {cache_dir.absolute()}")
print(f"Representation shape: {representation.shape}")
print(f"Records created: {len(ds)}")
print(f"Layer: {layer_name}")

print("\n" + "=" * 70)
