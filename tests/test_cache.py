"""Unit tests for src/data/cache.py"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from src.data.cache import RepresentationCache


def test_cache_initialization():
    """Test cache initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        assert cache.cache_dir.exists()
        assert cache.dtype == np.float16


def test_cache_fp32_vs_fp16():
    """Test fp32 vs fp16 dtype."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_fp32 = RepresentationCache(Path(tmpdir) / "fp32", use_fp16=False)
        cache_fp16 = RepresentationCache(Path(tmpdir) / "fp16", use_fp16=True)

        assert cache_fp32.dtype == np.float32
        assert cache_fp16.dtype == np.float16


def test_save_metadata():
    """Test metadata saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        layer_name = "test"
        layer_dir = cache.get_layer_path(layer_name)
        layer_dir.mkdir(parents=True, exist_ok=True)

        metadata = [{"id": 0, "prompt": "test"}]
        cache._save_metadata_to_process_file(layer_name, metadata)

        metadata_file = layer_dir / f"metadata_process_{cache.pid}.json"
        assert metadata_file.exists()


def test_initialize_layer():
    """Test layer initialization creates correct memmap shape."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        layer_name = "test_layer"
        total_samples = 100
        feature_dim = 64

        memmap_array = cache.initialize_layer(layer_name, total_samples, feature_dim)

        assert memmap_array.shape == (total_samples, feature_dim)
        assert memmap_array.dtype == np.float16
        assert (cache.get_layer_path(layer_name) / "data.npy").exists()
        assert (cache.get_layer_path(layer_name) / "metadata.json").exists()


def test_atomic_counter():
    """Test atomic counter increments correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        layer_name = "test_layer"

        layer_dir = cache.get_layer_path(layer_name)
        layer_dir.mkdir(parents=True, exist_ok=True)

        # First increment should return 0
        start_idx_1 = cache._atomic_increment_counter(layer_name, 10)
        assert start_idx_1 == 0

        # Second increment should return 10
        start_idx_2 = cache._atomic_increment_counter(layer_name, 5)
        assert start_idx_2 == 10


def test_save_representation():
    """Test saving a representation tensor to cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        layer_name = "test_layer"

        # Initialize layer
        n_timesteps, n_spatial, n_features = 3, 8, 64
        total_samples = n_timesteps * n_spatial * 10
        cache.initialize_layer(layer_name, total_samples, n_features)

        # Create test representation [timesteps, batch, spatial, features]
        representation = torch.randn(n_timesteps, 1, n_spatial, n_features)

        cache.save_representation(
            layer_name=layer_name,
            object_name="cat",
            style="photo",
            prompt_nr=1,
            prompt_text="a photo of a cat",
            representation=representation,
            num_steps=50,
            guidance_scale=7.5,
        )

        # Verify data was written
        layer_info = cache._active_memmaps[layer_name]
        memmap_array = layer_info["memmap"]
        assert memmap_array[: n_timesteps * n_spatial].any()
