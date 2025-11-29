#!/usr/bin/env python3
"""
NPY memmap-based cache for fast data loading.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch


class RepresentationCache:
    """
    Fast numpy memmap cache for representations.
    Supports incremental writes without loading everything to RAM.
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

        self._active_memmaps: Dict[str, Dict] = {}

    def get_layer_path(self, layer_name: str) -> Path:
        """Get storage path for a layer."""
        return self.cache_dir / f"{layer_name.lower()}"

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
        info_path = layer_dir / "info.json"

        # Check if layer already exists
        if info_path.exists():
            print(f"⚠️  Layer {layer_name} already exists - skipping initialization")
            # Load and return existing memmap
            with open(info_path, "r") as f:
                info = json.load(f)

            memmap_array = np.memmap(
                str(data_path),
                dtype=self.dtype,
                mode="r+",
                shape=(info["total_samples"], info["feature_dim"]),
            )

            self._active_memmaps[layer_name] = {
                "memmap": memmap_array,
                "current_idx": info.get("current_count", 0),
                "metadata": [],
                "existing_metadata": info.get("metadata", []),
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
            "current_idx": 0,
            "metadata": [],
            "existing_metadata": [],
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
        info_path = layer_dir / "info.json"
        with open(info_path, "w") as f:
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

        # Expect shape: [timesteps, batch, spatial, features]
        if tensor.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [timesteps, batch, spatial, features], "
                f"got shape: {tensor.shape}"
            )

        tensor = tensor.squeeze(1)
        n_timesteps, n_spatial, n_features = tensor.shape

        total_records = n_timesteps * n_spatial
        flat_data = tensor.reshape(total_records, n_features).numpy()

        # Get or create memmap
        if layer_name not in self._active_memmaps:
            layer_dir = self.get_layer_path(layer_name)
            info_path = layer_dir / "info.json"

            if info_path.exists():
                # Load existing memmap
                with open(info_path, "r") as f:
                    info = json.load(f)

                data_path = layer_dir / "data.npy"
                memmap_array = np.memmap(
                    str(data_path),
                    dtype=self.dtype,
                    mode="r+",
                    shape=(info["total_samples"], info["feature_dim"]),
                )

                # Start from current_count
                current_idx = info.get("current_count", 0)

                # Load existing metadata from info.json
                existing_metadata = info.get("metadata", [])

                self._active_memmaps[layer_name] = {
                    "memmap": memmap_array,
                    "current_idx": current_idx,
                    "metadata": [],
                    "existing_metadata": existing_metadata,  # Preserve old metadata
                    "total_samples": info["total_samples"],
                    "feature_dim": info["feature_dim"],
                }
            else:
                raise RuntimeError(
                    f"Layer {layer_name} not initialized. "
                )

        layer_info = self._active_memmaps[layer_name]
        memmap_array = layer_info["memmap"]
        start_idx = layer_info["current_idx"]

        if start_idx + total_records > layer_info["total_samples"]:
            # Auto-resize memmap if out of space
            print("\n⚠️  Memmap full - resizing to accommodate new data...")
            old_size = layer_info["total_samples"]
            # Add 50% more space to avoid frequent resizes
            new_size = old_size + max(total_records, old_size // 2)
            print(f"   Resizing: {old_size:,} → {new_size:,} samples")

            layer_dir = self.get_layer_path(layer_name)
            data_path = layer_dir / "data.npy"

            # Close current memmap
            memmap_array.flush()
            del memmap_array

            # Create larger memmap
            old_data = np.memmap(
                str(data_path),
                dtype=self.dtype,
                mode="r",
                shape=(old_size, layer_info["feature_dim"]),
            )

            temp_path = data_path.with_suffix(".tmp")
            new_memmap = np.memmap(
                str(temp_path),
                dtype=self.dtype,
                mode="w+",
                shape=(new_size, layer_info["feature_dim"]),
            )

            # Copy existing data
            print(f"   Copying {old_size:,} existing samples...")
            chunk_size = 10000
            for chunk_start in range(0, old_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, old_size)
                new_memmap[chunk_start:chunk_end] = old_data[chunk_start:chunk_end]

            new_memmap.flush()
            del new_memmap
            del old_data

            # Replace old file
            temp_path.replace(data_path)

            # Reload with new size
            memmap_array = np.memmap(
                str(data_path),
                dtype=self.dtype,
                mode="r+",
                shape=(new_size, layer_info["feature_dim"]),
            )

            layer_info["memmap"] = memmap_array
            layer_info["total_samples"] = new_size

            # Update info.json
            info_path = layer_dir / "info.json"
            with open(info_path, "r") as f:
                info = json.load(f)
            info["total_samples"] = new_size
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)

            print("   ✓ Resize complete")

        # Write data to memmap
        end_idx = start_idx + total_records
        memmap_array[start_idx:end_idx] = flat_data.astype(self.dtype)
        memmap_array.flush()

        layer_info["current_idx"] = end_idx

        # Update info.json with current progress
        layer_dir = self.get_layer_path(layer_name)
        info_path = layer_dir / "info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                info = json.load(f)
            info["current_count"] = end_idx
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)

        # Append metadata to in-memory list
        for i in range(n_timesteps):
            for j in range(n_spatial):
                layer_info["metadata"].append(
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

    def finalize_layer(self, layer_name: str):
        """
        Finalize a layer (trim memmap to actual size, close files).
        """
        if layer_name not in self._active_memmaps:
            print(f"Layer {layer_name} not active, skipping finalization")
            return

        layer_info = self._active_memmaps[layer_name]
        actual_size = layer_info["current_idx"]
        expected_size = layer_info["total_samples"]

        layer_dir = self.get_layer_path(layer_name)
        data_path = layer_dir / "data.npy"

        if actual_size < expected_size:
            print(f"Trimming {layer_name}: {expected_size} → {actual_size} samples")

            # Load full memmap
            old_memmap = layer_info["memmap"]
            old_memmap.flush()
            del old_memmap  # Close

            # Create trimmed version
            old_data = np.memmap(
                str(data_path),
                dtype=self.dtype,
                mode="r",
                shape=(expected_size, layer_info["feature_dim"]),
            )

            temp_path = data_path.with_suffix(".tmp")
            new_memmap = np.memmap(
                str(temp_path),
                dtype=self.dtype,
                mode="w+",
                shape=(actual_size, layer_info["feature_dim"]),
            )

            # Copy data in chunks to avoid OOM
            chunk_size = 10000
            for start_idx in range(0, actual_size, chunk_size):
                end_idx = min(start_idx + chunk_size, actual_size)
                new_memmap[start_idx:end_idx] = old_data[start_idx:end_idx]

            new_memmap.flush()
            del new_memmap
            del old_data

            temp_path.replace(data_path)

            # Update info
            info_path = layer_dir / "info.json"
            with open(info_path, "r") as f:
                info = json.load(f)
            info["total_samples"] = actual_size
            info["current_count"] = actual_size
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
        else:
            print(f"No trimming needed for {layer_name} (using {actual_size} samples)")
            memmap_array = layer_info["memmap"]
            memmap_array.flush()
            del memmap_array

        print(f"  Saving metadata for {layer_name}...")

        # Merge existing metadata with new entries
        existing_metadata = layer_info.get("existing_metadata", [])
        new_metadata = layer_info["metadata"]

        if existing_metadata:
            all_metadata = existing_metadata + new_metadata
            print(f"    Merged {len(existing_metadata)} existing + {len(new_metadata)} new entries")
        else:
            all_metadata = new_metadata

        # Update info.json with all metadata
        info_path = layer_dir / "info.json"
        with open(info_path, "r") as f:
            info = json.load(f)
        info["metadata"] = all_metadata
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"    Saved {len(all_metadata)} metadata entries to info.json")

        del self._active_memmaps[layer_name]

        print(f"✓ Finalized {layer_name}: {actual_size} samples")

    def check_exists(self, layer_name: str, object_name: str, style: str, prompt_nr: int) -> bool:
        """
        Check if a specific representation exists.
        Uses info.json metadata for lookups.

        Args:
            layer_name: Name of the layer
            object_name: Object name
            style: Style name
            prompt_nr: Prompt number

        Returns:
            bool: True if representation exists
        """
        layer_dir = self.get_layer_path(layer_name)
        info_path = layer_dir / "info.json"

        if not info_path.exists():
            return False

        with open(info_path, "r") as f:
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
