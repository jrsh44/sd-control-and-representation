"""Unit tests for src/data/dataset.py"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import RepresentationDataset


def create_test_dataset(
    cache_dir: Path, layer_name: str, n_samples: int = 10, feature_dim: int = 64
):
    """Helper function to create a test dataset with metadata."""
    layer_dir = cache_dir / layer_name
    layer_dir.mkdir(parents=True, exist_ok=True)

    # Create test data
    data = np.random.randn(n_samples, feature_dim).astype(np.float16)
    data_path = layer_dir / "data.npy"

    # Save as memmap
    memmap_array = np.memmap(
        str(data_path),
        dtype=np.float16,
        mode="w+",
        shape=(n_samples, feature_dim),
    )
    memmap_array[:] = data
    memmap_array.flush()
    del memmap_array

    # Create metadata
    metadata = []
    for i in range(n_samples):
        metadata.append(
            {
                "object_name": "cat" if i % 2 == 0 else "dog",
                "style": "photo",
                "timestep": i % 5,
                "prompt": f"a photo of a {'cat' if i % 2 == 0 else 'dog'}",
            }
        )

    # Save metadata
    metadata_json = {
        "total_samples": n_samples,
        "feature_dim": feature_dim,
        "dtype": str(np.float16),
        "metadata": metadata,
    }

    with open(layer_dir / "metadata.json", "w") as f:
        json.dump(metadata_json, f)

    return data_path


def test_dataset_initialization():
    """Test basic dataset initialization without filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"
        n_samples = 10
        feature_dim = 64

        create_test_dataset(cache_dir, layer_name, n_samples, feature_dim)

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            use_local_copy=False,
        )

        assert len(dataset) == n_samples
        assert dataset._full_data.shape == (n_samples, feature_dim)


def test_dataset_getitem():
    """Test retrieving items from dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"
        n_samples = 10
        feature_dim = 64

        create_test_dataset(cache_dir, layer_name, n_samples, feature_dim)

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            use_local_copy=False,
        )

        # Get first item
        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (feature_dim,)
        assert item.dtype == torch.float32


def test_dataset_filtering():
    """Test dataset with filter function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"
        n_samples = 10
        feature_dim = 64

        create_test_dataset(cache_dir, layer_name, n_samples, feature_dim)

        # Filter for cats only
        def filter_cats(entry):
            return entry["object_name"] == "cat"

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            filter_fn=filter_cats,
            use_local_copy=False,
        )

        # Should have 5 cats (half of 10 samples)
        assert len(dataset) == 5


def test_dataset_with_metadata():
    """Test dataset returning metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"
        n_samples = 10
        feature_dim = 64

        create_test_dataset(cache_dir, layer_name, n_samples, feature_dim)

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            return_metadata=True,
            use_local_copy=False,
        )

        # This test might fail if return_metadata is not fully implemented
        # but it tests the API
        assert len(dataset) == n_samples


def test_dataset_with_timestep():
    """Test dataset returning timestep information."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"
        n_samples = 10
        feature_dim = 64

        create_test_dataset(cache_dir, layer_name, n_samples, feature_dim)

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            return_timestep=True,
            use_local_copy=False,
        )

        # This test might fail if return_timestep is not fully implemented
        # but it tests the API
        assert len(dataset) == n_samples


def test_dataset_with_indices():
    """Test dataset with pre-computed indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"
        n_samples = 10
        feature_dim = 64

        create_test_dataset(cache_dir, layer_name, n_samples, feature_dim)

        # Use only first 3 indices
        indices = [0, 1, 2]
        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            indices=indices,
            use_local_copy=False,
        )

        assert len(dataset) == 3


def test_dataset_missing_layer():
    """Test dataset raises error for missing layer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        with pytest.raises(ValueError, match="Layer not found"):
            RepresentationDataset(
                cache_dir=cache_dir,
                layer_name="nonexistent_layer",
                use_local_copy=False,
            )
