#!/usr/bin/env python3
"""
Simple script to test loading cached representations with PyTorch DataLoader.

Usage:
    # First, create cache:
    uv run scripts/tests/create_representation.py --prompt "a cat" --steps 10
    uv run scripts/tests/cache_representation.py

    # Then test DataLoader:
    uv run scripts/tests/use_dataloader.py
"""

import sys
from pathlib import Path

from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import RepresentationDataset  # noqa: E402

print("=" * 70)
print("Testing DataLoader with Cached Representations")
print("=" * 70)

# Setup
cache_dir = Path("test_cache_output")
layer_name = "down_blocks.2.attentions.1"
dataset_path = cache_dir / layer_name

if not dataset_path.exists():
    print("\n❌ ERROR: Dataset directory not found!")
    print("Please run first:")
    print("  uv run scripts/tests/test_cache_simple.py")
    sys.exit(1)

# Load dataset
print(f"\n1. Loading dataset from: {dataset_path}")
dataset = RepresentationDataset(
    dataset_path=dataset_path,
    return_metadata=True,
)
print(f"   ✅ Loaded {len(dataset)} samples")

# Create DataLoader with shuffling
print("\n2. Creating DataLoader...")
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)
print("   Batch size: 32")
print("   Shuffle: True")
print(f"   Batches: {len(dataloader)}")

# Test iteration
print("\n3. Testing iteration (first 3 batches)...")
for i, batch in enumerate(dataloader):
    if i >= 3:
        break

    # Unpack: batch is a tuple of (features, metadata)
    features, metadata = batch

    print(f"\n   Batch {i + 1}:")
    print(f"     Features shape: {features.shape}")
    timestep_min = metadata["timestep"].min().item()
    timestep_max = metadata["timestep"].max().item()
    print(f"     Timestep range: {timestep_min}-{timestep_max}")
    spatial_min = metadata["spatial"].min().item()
    spatial_max = metadata["spatial"].max().item()
    print(f"     Spatial range: {spatial_min}-{spatial_max}")
    print(f"     Object: {metadata['object'][0]}")
    print(f"     Style: {metadata['style'][0]}")

# Test shuffling by comparing first batch across multiple epochs
print("\n4. Testing shuffle behavior...")
print("   Collecting first 5 samples from first batch across 3 epochs...")
print("   If shuffling works, the order should be DIFFERENT each time.\n")

epochs_data = []
for _ in range(3):
    # Create new iterator for each epoch
    for i, batch in enumerate(dataloader):
        if i == 0:  # Only first batch
            features, metadata = batch
            # Get first 5 samples info
            batch_info = []
            for j in range(min(5, features.shape[0])):
                info = (
                    metadata["timestep"][j].item(),
                    metadata["spatial"][j].item(),
                    metadata["object"][j],
                    metadata["style"][j],
                )
                batch_info.append(info)
            epochs_data.append(batch_info)
        break

# Display the data
for epoch_idx, epoch_data in enumerate(epochs_data):
    print(f"   Epoch {epoch_idx + 1} - First batch, first 5 samples:")
    for sample_idx, (timestep, spatial, obj, style) in enumerate(epoch_data):
        print(f"     [{sample_idx}] t={timestep:2d}, s={spatial:4d}, {obj:15s}, {style}")

# Check if any differences exist
print("\n   Analysis:")
if epochs_data[0] == epochs_data[1] == epochs_data[2]:
    print("     ⚠️  WARNING: All epochs have IDENTICAL order - shuffle may not be working!")
else:
    differences = 0
    if epochs_data[0] != epochs_data[1]:
        differences += 1
        print("     ✅ Epoch 1 vs Epoch 2: DIFFERENT order (shuffle working)")
    if epochs_data[0] != epochs_data[2]:
        differences += 1
        print("     ✅ Epoch 1 vs Epoch 3: DIFFERENT order (shuffle working)")
    if epochs_data[1] != epochs_data[2]:
        differences += 1
        print("     ✅ Epoch 2 vs Epoch 3: DIFFERENT order (shuffle working)")

    print(f"\n     🎲 Shuffle verified! Found {differences}/3 different orderings.")

# Additional proof: Show which timesteps appear in position 0 across epochs
print("\n   Proof of shuffling - Sample at position [0] across epochs:")
for epoch_idx, epoch_data in enumerate(epochs_data):
    timestep, spatial, obj, style = epoch_data[0]
    print(f"     Epoch {epoch_idx + 1}: timestep={timestep}, spatial={spatial}")

unique_first_samples = len({epoch_data[0] for epoch_data in epochs_data})
print(f"\n     Unique samples in position [0]: {unique_first_samples}/3")
if unique_first_samples > 1:
    print("     ✅ Position [0] contains different samples = Shuffle confirmed!")
else:
    print("     ⚠️  Position [0] has same sample (might be coincidence with small dataset)")

# Summary
print("\n" + "=" * 70)
print("✅ DataLoader works correctly!")
print("=" * 70)
print(f"\nDataset size: {len(dataset)} samples")
print(f"Batches per epoch: {len(dataloader)}")

# Get first sample to show feature dimension
first_sample = dataset[0]
features_sample, _ = first_sample
print(f"Feature dimension: {features_sample.shape[0]}")
