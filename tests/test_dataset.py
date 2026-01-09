"""Unit tests for src/data/dataset.py"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import RepresentationDataset


def create_test_dataset(
    cache_dir: Path,
    layer_name: str,
    n_samples: int = 10,
    feature_dim: int = 64,
    entries_key: str = "entries",
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
                "prompt_nr": i,
                "start_idx": i,
                "end_idx": i + 1,
            }
        )

    # Save metadata
    metadata_json = {
        "total_samples": n_samples,
        "feature_dim": feature_dim,
        "dtype": str(np.float16),
        entries_key: metadata,
    }

    with open(layer_dir / "metadata.json", "w") as f:
        json.dump(metadata_json, f)

    return data_path


# =============================================================================
# Basic Initialization Tests
# =============================================================================


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

        dataset.close()


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

        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (feature_dim,)
        assert item.dtype == torch.float32

        dataset.close()


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


# =============================================================================
# Property Tests
# =============================================================================


def test_dataset_feature_dim_property():
    """Test feature_dim property."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"
        feature_dim = 128

        create_test_dataset(cache_dir, layer_name, 10, feature_dim)

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            use_local_copy=False,
        )

        assert dataset.feature_dim == feature_dim

        dataset.close()


# =============================================================================
# Filtering Tests
# =============================================================================


def test_dataset_filtering():
    """Test dataset with filter function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"
        n_samples = 10
        feature_dim = 64

        create_test_dataset(cache_dir, layer_name, n_samples, feature_dim)

        def filter_cats(entry):
            return entry["object_name"] == "cat"

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            filter_fn=filter_cats,
            use_local_copy=False,
        )

        assert len(dataset) == 5

        dataset.close()


def test_dataset_complex_filter():
    """Test dataset with complex filter function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"

        create_test_dataset(cache_dir, layer_name, n_samples=20)

        def complex_filter(entry):
            return entry["object_name"] == "cat" and entry["timestep"] < 3

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            filter_fn=complex_filter,
            use_local_copy=False,
        )

        assert len(dataset) < 10
        assert len(dataset) > 0

        dataset.close()


# =============================================================================
# Indexing Tests
# =============================================================================


def test_dataset_direct_indexing():
    """Test dataset uses direct indexing when no filter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"
        n_samples = 100

        create_test_dataset(cache_dir, layer_name, n_samples=n_samples)

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            use_local_copy=False,
        )

        assert dataset.use_direct_indexing is True
        assert dataset.indices is None
        assert len(dataset) == n_samples

        dataset.close()


def test_dataset_indirect_indexing_with_filter():
    """Test dataset uses indirect indexing with filter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"

        create_test_dataset(cache_dir, layer_name, n_samples=10)

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            filter_fn=lambda x: x["object_name"] == "cat",
            use_local_copy=False,
        )

        assert dataset.use_direct_indexing is False
        assert dataset.indices is not None
        assert len(dataset.indices) == 5

        dataset.close()


def test_dataset_with_indices():
    """Test dataset with pre-computed indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"
        n_samples = 10
        feature_dim = 64

        create_test_dataset(cache_dir, layer_name, n_samples, feature_dim)

        indices = [0, 1, 2]
        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            indices=indices,
            use_local_copy=False,
        )

        assert len(dataset) == 3

        dataset.close()


def test_dataset_empty_indices():
    """Test dataset with empty indices list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"

        create_test_dataset(cache_dir, layer_name, n_samples=10)

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            indices=[],
            use_local_copy=False,
        )

        assert len(dataset) == 0

        dataset.close()


def test_dataset_out_of_order_indices():
    """Test dataset with out-of-order indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"

        create_test_dataset(cache_dir, layer_name, n_samples=10)

        indices = [9, 5, 3, 1]
        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            indices=indices,
            use_local_copy=False,
        )

        assert len(dataset) == 4
        for i in range(len(dataset)):
            item = dataset[i]
            assert isinstance(item, torch.Tensor)

        dataset.close()


# =============================================================================
# Metadata and Transform Tests
# =============================================================================


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

        assert len(dataset) == n_samples

        dataset.close()


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

        assert len(dataset) == n_samples

        dataset.close()


def test_dataset_with_transform():
    """Test dataset with custom transform function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"

        create_test_dataset(cache_dir, layer_name)

        def double_transform(x):
            return x * 2

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            transform=double_transform,
            use_local_copy=False,
        )

        item = dataset[0]
        assert isinstance(item, torch.Tensor)

        dataset.close()


def test_dataset_metadata_key_fallback():
    """Test dataset can load from 'metadata' key instead of 'entries'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"

        create_test_dataset(cache_dir, layer_name, entries_key="metadata")

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            filter_fn=lambda x: x["object_name"] == "cat",
            use_local_copy=False,
        )

        assert len(dataset) == 5

        dataset.close()


# =============================================================================
# Data Type Tests
# =============================================================================


def test_dataset_fp32_dtype():
    """Test dataset with fp32 data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"
        n_samples = 10
        feature_dim = 64

        layer_dir = cache_dir / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        data = np.random.randn(n_samples, feature_dim).astype(np.float32)
        data_path = layer_dir / "data.npy"

        memmap_array = np.memmap(
            str(data_path),
            dtype=np.float32,
            mode="w+",
            shape=(n_samples, feature_dim),
        )
        memmap_array[:] = data
        memmap_array.flush()
        del memmap_array

        metadata_json = {
            "total_samples": n_samples,
            "feature_dim": feature_dim,
            "dtype": "<class 'numpy.float32'>",
            "entries": [],
        }

        with open(layer_dir / "metadata.json", "w") as f:
            json.dump(metadata_json, f)

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            use_local_copy=False,
        )

        assert dataset._full_data.dtype == np.float32

        dataset.close()


def test_dataset_large_feature_dim():
    """Test dataset with large feature dimension."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"
        feature_dim = 2048

        create_test_dataset(cache_dir, layer_name, n_samples=5, feature_dim=feature_dim)

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            use_local_copy=False,
        )

        item = dataset[0]
        assert item.shape == (feature_dim,)

        dataset.close()


def test_dataset_missing_metadata_with_no_filter():
    """Test dataset can load without metadata when no filter needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        layer_name = "test_layer"
        n_samples = 10
        feature_dim = 64

        layer_dir = cache_dir / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        data = np.random.randn(n_samples, feature_dim).astype(np.float16)
        data_path = layer_dir / "data.npy"

        memmap_array = np.memmap(
            str(data_path),
            dtype=np.float16,
            mode="w+",
            shape=(n_samples, feature_dim),
        )
        memmap_array[:] = data
        memmap_array.flush()
        del memmap_array

        metadata_json = {
            "total_samples": n_samples,
            "feature_dim": feature_dim,
            "dtype": str(np.float16),
        }

        with open(layer_dir / "metadata.json", "w") as f:
            json.dump(metadata_json, f)

        dataset = RepresentationDataset(
            cache_dir=cache_dir,
            layer_name=layer_name,
            use_local_copy=False,
        )

        assert len(dataset) == n_samples

        dataset.close()
