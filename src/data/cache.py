#!/usr/bin/env python3
"""
Class for fast numpy memmap cache for representations.
"""

import atexit
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch


class RepresentationCache:
    """
    Fast numpy memmap cache for representations.
    Multi-process safe: uses atomic counter + per-process metadata files.
    """

    def __init__(self, cache_dir: Path, use_fp16: bool = True):
        """
        Args:
            cache_dir: Base directory for cache
            use_fp16: Store as float16 to save space
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_fp16 = use_fp16
        self.dtype = np.float16 if use_fp16 else np.float32

        # Track active memmap files for each layer
        self._active_memmaps: Dict[str, Dict] = {}

        # Process ID for unique metadata files
        self.pid = os.getpid()

        self._register_cleanup_handlers()

    def _register_cleanup_handlers(self):
        """
        Register handlers to save metadata on unexpected exit.
        """

        def cleanup_handler(signum=None, frame=None):
            print(f"\n⚠️  Process {self.pid} interrupted (signal={signum}), saving metadata...")
            self._emergency_save_all_metadata()
            sys.exit(1 if signum else 0)

        signal.signal(signal.SIGTERM, cleanup_handler)
        signal.signal(signal.SIGINT, cleanup_handler)
        atexit.register(lambda: self._emergency_save_all_metadata())

    def _emergency_save_all_metadata(self):
        """
        Save all pending metadata from all layers.
        """

        for layer_name, layer_info in self._active_memmaps.items():
            try:
                if layer_info["metadata"]:
                    self._save_metadata_to_process_file(layer_name, layer_info["metadata"])
                    metadata_count = len(layer_info["metadata"])
                    print(f"  ✓ Saved {metadata_count} metadata entries for {layer_name}")
            except Exception as e:
                print(f"  ✗ Failed to save metadata for {layer_name}: {e}")

    def _save_metadata_to_process_file(self, layer_name: str, metadata_list: list):
        """
        Save metadata to process-specific file

        Args:
            layer_name: Layer name
            metadata_list: List of metadata entries to save
        """

        layer_dir = self.get_layer_path(layer_name)
        metadata_file = layer_dir / f"metadata_process_{self.pid}.json"

        existing = []
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                existing = json.load(f)

        all_metadata = existing + metadata_list

        temp_file = metadata_file.with_suffix(f".tmp.{self.pid}")
        with open(temp_file, "w") as f:
            json.dump(all_metadata, f, indent=2)

        temp_file.replace(metadata_file)

    def get_layer_path(self, layer_name: str) -> Path:
        """
        Get storage path for a layer.

        Args:
            layer_name: Layer name

        Returns:
            Path to layer directory
        """
        return self.cache_dir / f"{layer_name.lower()}"

    def _atomic_increment_counter(self, layer_name: str, count: int) -> int:
        """
        Atomically reserve `count` indices and return starting index.
        Thread-safe across multiple processes using atomic file rename.

        Args:
            layer_name: Layer name
            count: Number of indices to reserve

        Returns:
            Starting index for this process
        """
        layer_dir = self.get_layer_path(layer_name)
        counter_file = layer_dir / "counter.txt"

        max_retries = 100
        for attempt in range(max_retries):
            try:
                if counter_file.exists():
                    with open(counter_file, "r") as f:
                        current = int(f.read().strip())
                else:
                    current = 0

                next_value = current + count

                temp_file = counter_file.with_suffix(f".tmp.{self.pid}.{attempt}")
                with open(temp_file, "w") as f:
                    f.write(str(next_value))

                try:
                    temp_file.replace(counter_file)
                    return current
                except OSError:
                    temp_file.unlink(missing_ok=True)
                    time.sleep(0.001)
                    continue

            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to get atomic index after {max_retries} attempts: {e}"
                    ) from e
                time.sleep(0.001)
                continue

        raise RuntimeError("Unreachable")

    def initialize_layer(self, layer_name: str, total_samples: int, feature_dim: int) -> np.memmap:
        """
        Initialize memmap file for a layer.

        Args:
            layer_name: Name of the layer
            total_samples: Total number of samples that will be written
            feature_dim: Dimension of features

        Returns:
            np.memmap array that can be written to incrementally
        """
        layer_dir = self.get_layer_path(layer_name)
        layer_dir.mkdir(parents=True, exist_ok=True)

        data_path = layer_dir / "data.npy"
        metadata_path = layer_dir / "metadata.json"

        # Check if layer already exists
        if metadata_path.exists():
            print(f"⚠️  Layer {layer_name} already exists - skipping initialization")
            with open(metadata_path, "r") as f:
                info = json.load(f)

            memmap_array = np.memmap(
                str(data_path),
                dtype=self.dtype,
                mode="r+",
                shape=(info["total_samples"], info["feature_dim"]),
            )

            self._active_memmaps[layer_name] = {
                "memmap": memmap_array,
                "metadata": [],
                "total_samples": info["total_samples"],
                "feature_dim": info["feature_dim"],
            }

            print(
                f"Loaded existing cache: {data_path} "
                f"[{info['total_samples']} x {info['feature_dim']}]"
            )
            return memmap_array

        # Create new memmap file
        memmap_array = np.memmap(
            str(data_path),
            dtype=self.dtype,
            mode="w+",
            shape=(total_samples, feature_dim),
        )

        # Track active memmap
        self._active_memmaps[layer_name] = {
            "memmap": memmap_array,
            "metadata": [],
            "total_samples": total_samples,
            "feature_dim": feature_dim,
        }

        # Save layer info with metadata
        info = {
            "total_samples": total_samples,
            "feature_dim": feature_dim,
            "dtype": str(self.dtype),
            "use_fp16": self.use_fp16,
            "current_count": 0,
            "metadata": [],
        }
        metadata_path = layer_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(info, f, indent=2)

        print(f"Initialized NPY cache: {data_path} [{total_samples} x {feature_dim}]")
        return memmap_array

    def save_representation(
        self,
        layer_name: str,
        object_name: str,
        style: str,
        prompt_nr: int,
        prompt_text: str,
        representation: torch.Tensor,
        num_steps: int,
        guidance_scale: float,
    ):
        """
        Save a single representation to memmap cache.

        Args:
            layer_name: Name of the layer
            object_name: Object name
            style: Style name
            prompt_nr: Prompt number
            prompt_text: Full prompt text
            representation: Tensor with shape [timesteps, batch, spatial, features]
            num_steps: Number of inference steps used
            guidance_scale: Guidance scale used
        """
        tensor = representation.cpu().half() if self.use_fp16 else representation.cpu()

        if tensor.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [timesteps, batch, spatial, features], "
                f"got shape: {tensor.shape}"
            )

        tensor = tensor.squeeze(1)
        n_timesteps, n_spatial, n_features = tensor.shape

        total_records = n_timesteps * n_spatial
        flat_data = tensor.reshape(total_records, n_features).numpy()

        if layer_name not in self._active_memmaps:
            layer_dir = self.get_layer_path(layer_name)
            metadata_path = layer_dir / "metadata.json"

            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    info = json.load(f)

                data_path = layer_dir / "data.npy"
                memmap_array = np.memmap(
                    str(data_path),
                    dtype=self.dtype,
                    mode="r+",
                    shape=(info["total_samples"], info["feature_dim"]),
                )

                self._active_memmaps[layer_name] = {
                    "memmap": memmap_array,
                    "metadata": [],
                    "total_samples": info["total_samples"],
                    "feature_dim": info["feature_dim"],
                }
            else:
                raise RuntimeError(f"Layer {layer_name} not initialized.")

        layer_info = self._active_memmaps[layer_name]
        memmap_array = layer_info["memmap"]

        start_idx = self._atomic_increment_counter(layer_name, total_records)

        if start_idx + total_records > layer_info["total_samples"]:
            self._resize_memmap(layer_name, start_idx + total_records)
            memmap_array = self._active_memmaps[layer_name]["memmap"]

        end_idx = start_idx + total_records
        memmap_array[start_idx:end_idx] = flat_data.astype(self.dtype)
        memmap_array.flush()

        new_metadata = []
        for i in range(n_timesteps):
            for j in range(n_spatial):
                new_metadata.append(
                    {
                        "timestep": int(i),
                        "spatial": int(j),
                        "object": object_name,
                        "style": style,
                        "prompt_nr": int(prompt_nr),
                        "prompt_text": prompt_text,
                        "num_steps": int(num_steps),
                        "guidance_scale": float(guidance_scale),
                    }
                )

        layer_info["metadata"].extend(new_metadata)

        if len(layer_info["metadata"]) >= 10:
            self._save_metadata_to_process_file(layer_name, layer_info["metadata"])
            layer_info["metadata"] = []

    def _resize_memmap(self, layer_name: str, min_size: int):
        """Resize memmap to accommodate more data with advisory locking."""
        layer_info = self._active_memmaps[layer_name]
        layer_dir = self.get_layer_path(layer_name)
        data_path = layer_dir / "data.npy"
        metadata_path = layer_dir / "metadata.json"
        lock_file = layer_dir / "resize.lock"

        lock_fd = open(lock_file, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            with open(metadata_path, "r") as f:
                info = json.load(f)

            current_size = info["total_samples"]

            if min_size <= current_size:
                print(f"  ℹ️  Another process already resized to {current_size}")
                layer_info["memmap"].flush()
                del layer_info["memmap"]

                layer_info["memmap"] = np.memmap(
                    str(data_path),
                    dtype=self.dtype,
                    mode="r+",
                    shape=(current_size, layer_info["feature_dim"]),
                )
                layer_info["total_samples"] = current_size
                return

            new_size = max(min_size, int(current_size * 1.5))
            print(f"  🔧 Resizing {layer_name}: {current_size:,} → {new_size:,}")

            layer_info["memmap"].flush()
            del layer_info["memmap"]

            old_data = np.memmap(
                str(data_path),
                dtype=self.dtype,
                mode="r",
                shape=(current_size, layer_info["feature_dim"]),
            )

            temp_path = data_path.with_suffix(f".tmp.{self.pid}")
            new_memmap = np.memmap(
                str(temp_path),
                dtype=self.dtype,
                mode="w+",
                shape=(new_size, layer_info["feature_dim"]),
            )

            chunk_size = 10000
            for chunk_start in range(0, current_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, current_size)
                new_memmap[chunk_start:chunk_end] = old_data[chunk_start:chunk_end]

            new_memmap.flush()
            del new_memmap
            del old_data

            temp_path.replace(data_path)

            info["total_samples"] = new_size
            with open(metadata_path, "w") as f:
                json.dump(info, f, indent=2)

            layer_info["memmap"] = np.memmap(
                str(data_path),
                dtype=self.dtype,
                mode="r+",
                shape=(new_size, layer_info["feature_dim"]),
            )
            layer_info["total_samples"] = new_size

            print("  ✓ Resize complete")

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def finalize_layer(self, layer_name: str):
        """Finalize layer by merging all process-specific metadata files."""
        if layer_name not in self._active_memmaps:
            print(f"Layer {layer_name} not active, skipping finalization")
            return

        layer_info = self._active_memmaps[layer_name]
        layer_dir = self.get_layer_path(layer_name)

        if layer_info["metadata"]:
            self._save_metadata_to_process_file(layer_name, layer_info["metadata"])
            layer_info["metadata"] = []

        layer_info["memmap"].flush()
        del layer_info["memmap"]

        print(f"  📦 Merging metadata from all processes for {layer_name}...")
        all_metadata = []

        metadata_files = sorted(layer_dir.glob("metadata_process_*.json"))
        for metadata_file in metadata_files:
            with open(metadata_file, "r") as f:
                process_metadata = json.load(f)
                all_metadata.extend(process_metadata)
            print(f"    Loaded {len(process_metadata)} entries from {metadata_file.name}")

        counter_file = layer_dir / "counter.txt"
        actual_size = 0
        if counter_file.exists():
            with open(counter_file, "r") as f:
                actual_size = int(f.read().strip())

        metadata_path = layer_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            info = json.load(f)

        info["metadata"] = all_metadata
        info["total_samples"] = actual_size
        info["current_count"] = actual_size

        with open(metadata_path, "w") as f:
            json.dump(info, f, indent=2)

        print(f"  ✓ Saved {len(all_metadata)} total metadata entries")
        print(f"  ✓ Finalized {layer_name}: {actual_size} samples")

        del self._active_memmaps[layer_name]

    def check_exists(self, layer_name: str, object_name: str, style: str, prompt_nr: int) -> bool:
        """
        Check if a representation already exists in the cache.

        Args:
            layer_name: Name of the layer
            object_name: Object name
            style: Style name
            prompt_nr: Prompt number

        Returns:
            bool: True if representation exists
        """
        layer_dir = self.get_layer_path(layer_name)
        metadata_path = layer_dir / "metadata.json"

        if not metadata_path.exists():
            return False

        with open(metadata_path, "r") as f:
            info = json.load(f)

        metadata = info.get("metadata", [])

        # Check if any metadata entry matches
        for meta in metadata:
            if (
                meta["object"] == object_name
                and meta["style"] == style
                and meta["prompt_nr"] == prompt_nr
            ):
                return True

        return False
