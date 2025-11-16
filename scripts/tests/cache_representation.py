#!/usr/bin/env python3
"""
Test memmap cache with a real representation from .pt file.

Usage:
    # First, create test representation:
    uv run scripts/tests/create_representation.py --prompt "a cat" --steps 10

    # Then, test cache with it:
    uv run scripts/tests/cache_representation.py
"""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.cache import RepresentationCache  # noqa: E402

print("=" * 70)
print("Testing Memmap Cache with Real Representation")
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
    print("   Run: uv run scripts/tests/create_representation.py --prompt \"a cat\" --steps 10")
    sys.exit(1)

# Initialize memmap cache
print("\n1. Initializing memmap cache...")
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

# Calculate total samples and initialize layer
total_samples = timesteps * spatial
print(f"\n3. Initializing layer (will create {total_samples} samples)...")
cache.initialize_layer(layer_name, total_samples=total_samples, feature_dim=features)

# Test 1: Save representation
print(f"\n4. Saving {layer_type} representation...")
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

# Test 2: Finalize layer
print("\n5. Finalizing layer...")
cache.finalize_layer(layer_name)
print("   ✅ Finalized")

# Test 3: Check files exist
print("\n6. Checking generated files...")
layer_path = cache_dir / layer_name.lower()
data_file = layer_path / "data.npy"
info_file = layer_path / "info.json"
meta_file = layer_path / "metadata.pkl"
index_file = layer_path / "index.json"

files_ok = True
for f, name in [(data_file, "data.npy"), (info_file, "info.json"), 
                (meta_file, "metadata.pkl"), (index_file, "index.json")]:
    if f.exists():
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   ✅ {name}: {size_mb:.2f} MB")
    else:
        print(f"   ❌ {name}: missing")
        files_ok = False

# Test 4: Check existence
print("\n7. Testing existence check...")
exists = cache.check_exists(layer_name, "cat", "Impressionism", 1)
print(f"   cat/Impressionism/1 exists: {exists} {'✅' if exists else '❌'}")

# Test 5: Load with memmap
print("\n8. Loading with memmap (zero RAM!)...")
data, metadata = cache.load_layer(layer_name, mmap_mode="r")
print(f"   Data shape: {data.shape}")
print(f"   Metadata entries: {len(metadata)}")

# Calculate expected records
expected_records = timesteps * spatial
print(f"   Expected: {expected_records} records ({timesteps} timesteps × {spatial} spatial)")

if data.shape[0] == expected_records:
    print("   ✅ Record count matches")
else:
    print(f"   ❌ Mismatch: expected {expected_records}, got {data.shape[0]}")

# Test 6: Check metadata structure
print("\n9. Checking metadata structure...")
if metadata:
    first_record = metadata[0]
    print("\n   First record:")
    print(f"     - timestep: {first_record['timestep']}")
    print(f"     - spatial: {first_record['spatial']}")
    print(f"     - object: {first_record['object']}")
    print(f"     - style: {first_record['style']}")
    print(f"     - prompt_text: {first_record['prompt_text'][:50]}...")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

success = exists and data.shape[0] == expected_records and files_ok

if success:
    print("✅ All tests passed!")
else:
    print("❌ Some tests failed!")

print(f"\nCache location: {cache_dir.absolute()}")
print(f"Representation shape: {representation.shape}")
print(f"Records created: {data.shape[0]}")
print(f"Layer: {layer_name}")
print(f"Storage: Memmap (zero RAM usage)")

print("\n" + "=" * 70)
