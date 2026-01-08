"""Unit tests for src/data/cache.py"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.data.cache import RepresentationCache


# =============================================================================
# Basic Initialization Tests
# =============================================================================


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


def test_cache_get_layer_path():
    """Test get_layer_path returns correct path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        layer_path = cache.get_layer_path("test_layer")

        assert layer_path == Path(tmpdir) / "test_layer"
        assert isinstance(layer_path, Path)


def test_cache_pid_tracking():
    """Test that cache tracks process ID correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        assert cache.pid == os.getpid()


# =============================================================================
# Array Job Tests
# =============================================================================


def test_cache_array_job_primary():
    """Test cache with primary array job (task 0)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(
            Path(tmpdir),
            use_fp16=True,
            array_id=0,
            array_total=4,
        )

        assert cache.array_id == 0
        assert cache.array_total == 4
        assert cache.is_primary is True


def test_cache_array_job_secondary():
    """Test cache with secondary array job (task > 0)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(
            Path(tmpdir),
            use_fp16=True,
            array_id=2,
            array_total=4,
        )

        assert cache.array_id == 2
        assert cache.array_total == 4
        assert cache.is_primary is False


def test_cache_no_array_job():
    """Test cache without array job (single process)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)

        assert cache.array_id is None
        assert cache.array_total is None
        assert cache.is_primary is True


# =============================================================================
# Metadata Tests
# =============================================================================


def test_save_metadata():
    """Test metadata saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        layer_name = "test"
        layer_dir = cache.get_layer_path(layer_name)
        layer_dir.mkdir(parents=True, exist_ok=True)

        metadata = [{"id": 0, "prompt": "test"}]
        cache._save_metadata_to_temp_file(layer_name, metadata)

        metadata_file = layer_dir / f"metadata_process_{cache.pid}.jsonl"
        assert metadata_file.exists()


# =============================================================================
# Layer Initialization Tests
# =============================================================================


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

        del memmap_array
        cache.close()


def test_initialize_layer_creates_metadata():
    """Test that initialize_layer creates metadata.json with correct info."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        layer_name = "test_layer"
        total_samples = 100
        feature_dim = 64

        cache.initialize_layer(layer_name, total_samples, feature_dim)

        metadata_path = cache.get_layer_path(layer_name) / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["total_samples"] == total_samples
        assert metadata["feature_dim"] == feature_dim
        assert "float16" in str(metadata["dtype"])

        cache.close()


def test_cache_multiple_layers():
    """Test cache can handle multiple layers simultaneously."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)

        layers = ["layer1", "layer2", "layer3"]
        for layer_name in layers:
            cache.initialize_layer(layer_name, 100, 64)

        for layer_name in layers:
            assert cache.get_layer_path(layer_name).exists()
            assert (cache.get_layer_path(layer_name) / "data.npy").exists()

        cache.close()


# =============================================================================
# Atomic Counter Tests
# =============================================================================


def test_atomic_counter():
    """Test atomic counter increments correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        layer_name = "test_layer"

        layer_dir = cache.get_layer_path(layer_name)
        layer_dir.mkdir(parents=True, exist_ok=True)

        start_idx_1 = cache._atomic_increment_counter(layer_name, 10)
        assert start_idx_1 == 0

        start_idx_2 = cache._atomic_increment_counter(layer_name, 5)
        assert start_idx_2 == 10


def test_atomic_counter_multiple_increments():
    """Test atomic counter with multiple sequential increments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        layer_name = "test_layer"

        layer_dir = cache.get_layer_path(layer_name)
        layer_dir.mkdir(parents=True, exist_ok=True)

        idx1 = cache._atomic_increment_counter(layer_name, 5)
        idx2 = cache._atomic_increment_counter(layer_name, 10)
        idx3 = cache._atomic_increment_counter(layer_name, 3)

        assert idx1 == 0
        assert idx2 == 5
        assert idx3 == 15


# =============================================================================
# Save Representation Tests
# =============================================================================


def test_save_representation():
    """Test saving a representation tensor to cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        layer_name = "test_layer"

        n_timesteps, n_spatial, n_features = 3, 8, 64
        total_samples = n_timesteps * n_spatial * 10
        cache.initialize_layer(layer_name, total_samples, n_features)

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

        layer_info = cache._active_memmaps[layer_name]
        memmap_array = layer_info["memmap"]
        assert memmap_array[: n_timesteps * n_spatial].any()

        del memmap_array, layer_info
        cache.close()


def test_save_representation_shape_handling():
    """Test save_representation handles different tensor shapes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        layer_name = "test_layer"

        n_timesteps, batch, n_spatial, n_features = 3, 1, 8, 64
        total_samples = n_timesteps * n_spatial * 10
        cache.initialize_layer(layer_name, total_samples, n_features)

        representation = torch.randn(n_timesteps, batch, n_spatial, n_features)

        cache.save_representation(
            layer_name=layer_name,
            object_name="test_obj",
            style="test_style",
            prompt_nr=1,
            prompt_text="test prompt",
            representation=representation,
            num_steps=50,
            guidance_scale=7.5,
        )

        assert layer_name in cache._active_memmaps

        cache.close()


def test_save_representation_metadata_content():
    """Test that save_representation creates correct metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        layer_name = "test_layer"

        n_timesteps, batch, n_spatial, n_features = 2, 1, 4, 32
        total_samples = n_timesteps * n_spatial * 10
        cache.initialize_layer(layer_name, total_samples, n_features)

        representation = torch.randn(n_timesteps, batch, n_spatial, n_features)

        cache.save_representation(
            layer_name=layer_name,
            object_name="cat",
            style="photo",
            prompt_nr=42,
            prompt_text="a photo of a cat",
            representation=representation,
            num_steps=50,
            guidance_scale=7.5,
        )

        layer_dir = cache.get_layer_path(layer_name)
        temp_files = list(layer_dir.glob("metadata_process_*.jsonl"))
        assert len(temp_files) > 0, "No temp metadata file created"

        with open(temp_files[0], "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 1

            entry = json.loads(lines[0])
            assert entry["prompt_nr"] == 42
            assert entry["object"] == "cat"
            assert entry["style"] == "photo"
            assert entry["n_timesteps"] == n_timesteps
            assert entry["n_spatial"] == n_spatial

        cache.close()


# =============================================================================
# Data Type Conversion Tests
# =============================================================================


def test_cache_dtype_conversion():
    """Test that cache correctly converts torch tensors to numpy dtype."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RepresentationCache(Path(tmpdir), use_fp16=True)
        layer_name = "test_layer"

        total_samples = 100
        feature_dim = 64
        memmap_array = cache.initialize_layer(layer_name, total_samples, feature_dim)

        test_tensor = torch.randn(10, feature_dim, dtype=torch.float32)
        memmap_array[:10] = test_tensor.numpy().astype(np.float16)

        assert memmap_array.dtype == np.float16

        del memmap_array
        cache.close()
