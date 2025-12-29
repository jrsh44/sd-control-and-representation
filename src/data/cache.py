#!/usr/bin/env python3
"""
Class for fast numpy memmap cache for representations.
"""

import atexit
import fcntl
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import torch


class RepresentationCache:
    """
    Numpy memmap cache for representations.

    Final output per layer:
    - data.npy: memmap file with all representations
    - metadata.json: all metadata entries + layer info

    During generation (temp files):
    - metadata_process_{pid}.jsonl: per-process metadata (merged on finalize/interrupt)
    """

    def __init__(
        self,
        cache_dir: Path,
        use_fp16: bool = True,
        array_id: Optional[int] = None,
        array_total: Optional[int] = None,
    ):
        """
        Args:
            cache_dir: Base directory for cache
            use_fp16: Store as float16 to save space
            array_id: SLURM array task ID (0-indexed), None if not array job
            array_total: Total number of array jobs, None if not array job
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_fp16 = use_fp16
        self.dtype = np.float16 if use_fp16 else np.float32

        # SLURM array job info
        self.array_id = array_id
        self.array_total = array_total
        self.is_primary = array_id is None or array_id == 0

        # Track active memmap files for each layer
        self._active_memmaps: Dict[str, Dict] = {}

        # Process ID for unique metadata files
        self.pid = os.getpid()
        self._cleanup_done = False

        self._register_cleanup_handlers()

    def _register_cleanup_handlers(self):
        """Register handlers to save metadata on unexpected exit."""

        def cleanup_handler(signum=None, frame=None):
            if self._cleanup_done:
                return
            self._cleanup_done = True

            signal_name = signal.Signals(signum).name if signum else "atexit"
            print(f"\n⚠️  Process {self.pid} interrupted ({signal_name}), saving metadata...")

            self._emergency_save_and_merge()

            if signum:
                sys.exit(128 + signum)

        signal.signal(signal.SIGTERM, cleanup_handler)
        signal.signal(signal.SIGINT, cleanup_handler)
        atexit.register(cleanup_handler)

    def _emergency_save_and_merge(self):
        """Save pending metadata and merge all temp files into metadata.json."""

        for layer_name, layer_info in self._active_memmaps.items():
            try:
                # Save any pending metadata to temp file
                if layer_info.get("pending_metadata"):
                    self._save_metadata_to_temp_file(layer_name, layer_info["pending_metadata"])
                    count = len(layer_info["pending_metadata"])
                    print(f"  ✓ Saved {count} pending entries for {layer_name}")
                    layer_info["pending_metadata"] = []

                if "memmap" in layer_info and layer_info["memmap"] is not None:
                    layer_info["memmap"].flush()

                self._merge_temp_metadata_files(layer_name)

            except Exception as e:
                print(f"  ✗ Failed to save/merge metadata for {layer_name}: {e}")

    def _save_metadata_to_temp_file(self, layer_name: str, metadata_list: list):
        """
        Append metadata to process-specific JSONL temp file.

        Args:
            layer_name: Layer name
            metadata_list: List of metadata entries to append
        """
        layer_dir = self.get_layer_path(layer_name)
        temp_file = layer_dir / f"metadata_process_{self.pid}.jsonl"

        with open(temp_file, "a") as f:
            for entry in metadata_list:
                json.dump(entry, f, separators=(",", ":"))
                f.write("\n")

    def _merge_temp_metadata_files(self, layer_name: str):
        """
        Merge all temp metadata files into the main metadata.json.

        Args:
            layer_name: Layer name
        """
        layer_dir = self.get_layer_path(layer_name)
        metadata_path = layer_dir / "metadata.json"
        lock_file = layer_dir / "merge.lock"

        lock_file.touch(exist_ok=True)

        lock_fd = open(lock_file, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    info = json.load(f)
            else:
                info = {}

            existing_entries = info.get("entries", [])
            # Use (prompt_nr, object, style) as unique key to distinguish entries across classes
            existing_keys = {
                (e["prompt_nr"], e.get("object", ""), e.get("style", "")) for e in existing_entries
            }

            new_entries = []
            temp_files = sorted(layer_dir.glob("metadata_process_*.jsonl"))

            for temp_file in temp_files:
                try:
                    with open(temp_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                entry = json.loads(line)
                                entry_key = (
                                    entry["prompt_nr"],
                                    entry.get("object", ""),
                                    entry.get("style", ""),
                                )
                                if entry_key not in existing_keys:
                                    new_entries.append(entry)
                                    existing_keys.add(entry_key)
                except Exception as e:
                    print(f"    ⚠️ Error reading {temp_file.name}: {e}")

            if new_entries:
                all_entries = existing_entries + new_entries
                info["entries"] = all_entries
                info["entry_count"] = len(all_entries)
                info["last_merge_timestamp"] = time.time()
                info["last_merge_pid"] = self.pid

                with open(metadata_path, "w") as f:
                    json.dump(info, f, indent=2)

                print(f"    ✓ Merged {len(new_entries)} new entries into metadata.json")

            for temp_file in temp_files:
                temp_file.unlink()

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def get_layer_path(self, layer_name: str) -> Path:
        """
        Get storage path for a layer.

        Args:
            layer_name: Name of the layer

        Returns:
            Path to the layer directory
        """
        return self.cache_dir / f"{layer_name.lower()}"

    def _atomic_increment_counter(self, layer_name: str, count: int) -> int:
        """
        Atomically reserve `count` indices and return starting index

        Args:
            layer_name: Name of the layer
            count: Number of indices to reserve

        Returns:
            Starting index for the reserved block
        """
        layer_dir = self.get_layer_path(layer_name)
        counter_file = layer_dir / "counter.txt"
        lock_file = layer_dir / "counter.lock"

        lock_file.touch(exist_ok=True)

        lock_fd = open(lock_file, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            if counter_file.exists():
                with open(counter_file, "r") as f:
                    current = int(f.read().strip())
            else:
                layer_dir = self.get_layer_path(layer_name)
                metadata_path = layer_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        info = json.load(f)
                        max_idx = 0
                        for entry in info.get("entries", []):
                            max_idx = max(max_idx, entry.get("end_idx", 0))
                        current = max_idx
                else:
                    current = 0

            next_value = current + count
            with open(counter_file, "w") as f:
                f.write(str(next_value))

            return current

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def _resize_memmap(self, layer_name: str, min_size: int):
        """
        Resize memmap to accommodate more data. Uses file locking for safety

        Args:
            layer_name: Name of the layer
            min_size: Minimum size needed after resize
        """
        layer_info = self._active_memmaps[layer_name]
        layer_dir = self.get_layer_path(layer_name)
        data_path = layer_dir / "data.npy"
        metadata_path = layer_dir / "metadata.json"
        lock_file = layer_dir / "resize.lock"

        lock_file.touch(exist_ok=True)
        lock_fd = open(lock_file, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            with open(metadata_path, "r") as f:
                info = json.load(f)

            current_size = info["total_samples"]

            if min_size <= current_size:
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

            # Read old data and create new larger file
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

            # Copy in chunks
            chunk_size = 10000
            for i in range(0, current_size, chunk_size):
                end = min(i + chunk_size, current_size)
                new_memmap[i:end] = old_data[i:end]

            new_memmap.flush()
            del new_memmap
            del old_data

            temp_path.replace(data_path)

            # Update metadata
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

    def initialize_layer(
        self,
        layer_name: str,
        total_samples: int,
        feature_dim: int,
        total_prompts: int = 0,
        samples_per_prompt: int = 0,
    ) -> np.memmap:
        """
        Initialize memmap file for a layer.

        Args:
            layer_name: Name of the layer
            total_samples: Total number of samples that will be written
            feature_dim: Dimension of features
            total_prompts: Total number of prompts across all jobs
            samples_per_prompt: Number of samples per prompt

        Returns:
            np.memmap array that can be written to
        """
        layer_dir = self.get_layer_path(layer_name)
        layer_dir.mkdir(parents=True, exist_ok=True)

        data_path = layer_dir / "data.npy"
        metadata_path = layer_dir / "metadata.json"
        init_lock = layer_dir / "init.lock"

        init_lock.touch(exist_ok=True)

        lock_fd = open(init_lock, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            if metadata_path.exists() and data_path.exists():
                with open(metadata_path, "r") as f:
                    info = json.load(f)

                if info["total_samples"] >= total_samples and info["feature_dim"] == feature_dim:
                    print(f"  ✓ Layer {layer_name} already initialized, opening existing")
                    memmap_array = np.memmap(
                        str(data_path),
                        dtype=self.dtype,
                        mode="r+",
                        shape=(info["total_samples"], info["feature_dim"]),
                    )

                    self._active_memmaps[layer_name] = {
                        "memmap": memmap_array,
                        "pending_metadata": [],
                        "total_samples": info["total_samples"],
                        "feature_dim": info["feature_dim"],
                        "samples_per_prompt": info.get("samples_per_prompt", samples_per_prompt),
                    }

                    return memmap_array

            print(f"  📦 Creating cache for {layer_name}: {total_samples:,} x {feature_dim}")

            memmap_array = np.memmap(
                str(data_path),
                dtype=self.dtype,
                mode="w+",
                shape=(total_samples, feature_dim),
            )

            info = {
                "total_samples": total_samples,
                "feature_dim": feature_dim,
                "dtype": str(self.dtype),
                "use_fp16": self.use_fp16,
                "total_prompts": total_prompts,
                "samples_per_prompt": samples_per_prompt,
                "created_timestamp": time.time(),
                "created_by_pid": self.pid,
                "entries": [],
                "entry_count": 0,
            }
            with open(metadata_path, "w") as f:
                json.dump(info, f, indent=2)

            counter_file = layer_dir / "counter.txt"
            with open(counter_file, "w") as f:
                f.write("0")

            self._active_memmaps[layer_name] = {
                "memmap": memmap_array,
                "pending_metadata": [],
                "total_samples": total_samples,
                "feature_dim": feature_dim,
                "samples_per_prompt": samples_per_prompt,
            }

            print(f"  ✓ Initialized {layer_name}: {data_path}")
            return memmap_array

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def get_existing_prompt_nrs(self, layer_name: str) -> Set[int]:
        """
        Get all prompt_nrs that already exist in the cache.
        Loads from metadata.json + any temp files for complete picture.

        Args:
            layer_name: Name of the layer

        Returns:
            Set of prompt_nr values that are already cached
        """
        layer_dir = self.get_layer_path(layer_name)
        existing: Set[int] = set()

        metadata_path = layer_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    info = json.load(f)
                for entry in info.get("entries", []):
                    existing.add(entry["prompt_nr"])
            except (json.JSONDecodeError, KeyError):
                pass

        for temp_file in layer_dir.glob("metadata_process_*.jsonl"):
            try:
                with open(temp_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            existing.add(entry["prompt_nr"])
            except (json.JSONDecodeError, KeyError):
                pass

        return existing

    def get_existing_entries(self, layer_name: str) -> Set[tuple]:
        """
        Get all (prompt_nr, object_name, style) tuples that already exist in the cache.
        Use this when prompt_nr alone is not unique (e.g., class-based generation).

        Args:
            layer_name: Name of the layer

        Returns:
            Set of (prompt_nr, object_name, style) tuples that are already cached
        """
        layer_dir = self.get_layer_path(layer_name)
        existing: Set[tuple] = set()

        metadata_path = layer_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    info = json.load(f)
                for entry in info.get("entries", []):
                    key = (entry["prompt_nr"], entry.get("object", ""), entry.get("style", ""))
                    existing.add(key)
            except (json.JSONDecodeError, KeyError):
                pass

        for temp_file in layer_dir.glob("metadata_process_*.jsonl"):
            try:
                with open(temp_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            key = (
                                entry["prompt_nr"],
                                entry.get("object", ""),
                                entry.get("style", ""),
                            )
                            existing.add(key)
            except (json.JSONDecodeError, KeyError):
                pass

        return existing

    def save_representation(
        self,
        layer_name: str,
        prompt_nr: int,
        prompt_text: str,
        representation: torch.Tensor,
        num_steps: int,
        guidance_scale: float,
        object_name: str = "",
        style: str = "",
    ):
        """
        Save a single representation tensor and its metadata

        Args:
            layer_name: Name of the layer
            prompt_nr: Prompt number
            prompt_text: Full prompt text
            representation: Tensor with shape [timesteps, batch, spatial, features]
            num_steps: Number of inference steps used
            guidance_scale: Guidance scale used
            object_name: Object name (optional)
            style: Style name (optional)
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

            if not metadata_path.exists():
                raise RuntimeError(
                    f"Layer {layer_name} not initialized. Call initialize_layer first."
                )

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
                "pending_metadata": [],
                "total_samples": info["total_samples"],
                "feature_dim": info["feature_dim"],
                "samples_per_prompt": info.get("samples_per_prompt", total_records),
            }

        layer_info = self._active_memmaps[layer_name]
        memmap_array = layer_info["memmap"]

        start_idx = self._atomic_increment_counter(layer_name, total_records)

        if start_idx + total_records > layer_info["total_samples"]:
            self._resize_memmap(layer_name, start_idx + total_records)
            memmap_array = self._active_memmaps[layer_name]["memmap"]

        end_idx = start_idx + total_records
        memmap_array[start_idx:end_idx] = flat_data.astype(self.dtype)
        memmap_array.flush()

        metadata_entry = {
            "prompt_nr": int(prompt_nr),
            "prompt_text": prompt_text,
            "object": object_name,
            "style": style,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "n_timesteps": int(n_timesteps),
            "n_spatial": int(n_spatial),
            "num_steps": int(num_steps),
            "guidance_scale": float(guidance_scale),
            "timestamp": time.time(),
        }

        self._save_metadata_to_temp_file(layer_name, [metadata_entry])

    def finalize_layer(self, layer_name: str):
        """
        Finalize layer by merging all temp metadata files into metadata.json.
        Also cleans up temp files after successful merge.

        Args:
            layer_name: Name of the layer
        """
        if layer_name not in self._active_memmaps:
            print(f"  ⚠️ Layer {layer_name} not active, attempting merge anyway...")
        else:
            layer_info = self._active_memmaps[layer_name]
            if layer_info.get("memmap") is not None:
                layer_info["memmap"].flush()

        self._merge_temp_metadata_files(layer_name)

        layer_dir = self.get_layer_path(layer_name)

        counter_file = layer_dir / "counter.txt"
        if counter_file.exists():
            counter_file.unlink()

        for lock_file in layer_dir.glob("*.lock"):
            lock_file.unlink()

        metadata_path = layer_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                info = json.load(f)
            info["finalized"] = True
            info["finalize_timestamp"] = time.time()
            with open(metadata_path, "w") as f:
                json.dump(info, f, indent=2)

        if layer_name in self._active_memmaps:
            del self._active_memmaps[layer_name]

        print(f"  ✓ Finalized {layer_name}")

    @classmethod
    def merge_all_process_files(cls, cache_dir: Path, layer_name: str) -> dict:
        """
        Standalone merge of all temp metadata files into metadata.json.

        Args:
            cache_dir: Base cache directory
            layer_name: Layer name to merge

        Returns:
            dict with merge statistics
        """
        layer_dir = Path(cache_dir) / layer_name.lower()

        if not layer_dir.exists():
            raise ValueError(f"Layer directory not found: {layer_dir}")

        metadata_path = layer_dir / "metadata.json"
        lock_file = layer_dir / "merge.lock"

        lock_file.touch(exist_ok=True)
        lock_fd = open(lock_file, "w")

        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    info = json.load(f)
            else:
                info = {"entries": []}

            existing_entries = info.get("entries", [])
            # Use (prompt_nr, object, style) as unique key to distinguish entries across classes
            existing_keys = {
                (e["prompt_nr"], e.get("object", ""), e.get("style", "")) for e in existing_entries
            }

            new_entries = []
            temp_files = sorted(layer_dir.glob("metadata_process_*.jsonl"))
            files_processed = 0

            for temp_file in temp_files:
                try:
                    with open(temp_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                entry = json.loads(line)
                                entry_key = (
                                    entry["prompt_nr"],
                                    entry.get("object", ""),
                                    entry.get("style", ""),
                                )
                                if entry_key not in existing_keys:
                                    new_entries.append(entry)
                                    existing_keys.add(entry_key)
                    files_processed += 1
                except Exception as e:
                    print(f"  ⚠️ Error reading {temp_file.name}: {e}")

            all_entries = existing_entries + new_entries
            info["entries"] = all_entries
            info["entry_count"] = len(all_entries)
            info["finalized"] = True
            info["merge_timestamp"] = time.time()

            with open(metadata_path, "w") as f:
                json.dump(info, f, indent=2)

            for temp_file in temp_files:
                temp_file.unlink()

            for f in layer_dir.glob("*.lock"):
                f.unlink()

            counter_file = layer_dir / "counter.txt"
            if counter_file.exists():
                counter_file.unlink()

            print(f"  ✓ Merged {len(new_entries)} entries from {files_processed} files")
            print(f"  ✓ Total entries: {len(all_entries)}")

            return {
                "layer": layer_name,
                "new_entries": len(new_entries),
                "total_entries": len(all_entries),
                "files_merged": files_processed,
            }

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()


def calculate_layer_dimensions(
    pipe,
    layer_paths: list,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    device: str = "cuda",
) -> Dict[str, Dict[str, int]]:
    """
    Run a single dummy inference to probe tensor dimensions for each layer.

    Args:
        pipe: Stable Diffusion pipeline
        layer_paths: List of LayerPath enum values to capture
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale
        device: Device to run on

    Returns:
        Dict mapping layer_name to dimension info
    """
    from src.models.sd_v1_5.hooks import capture_layer_representations

    print("🔍 Probing layer dimensions with dummy inference...")

    dummy_prompt = "a photo"
    generator = torch.Generator(device).manual_seed(42)

    representations, _ = capture_layer_representations(
        pipe=pipe,
        prompt=dummy_prompt,
        layer_paths=layer_paths,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    layer_dims = {}
    for layer_path, tensor in zip(layer_paths, representations, strict=True):
        if tensor is not None:
            n_timesteps = tensor.shape[0]
            n_spatial = tensor.shape[2]
            n_features = tensor.shape[3]
            samples_per_prompt = n_timesteps * n_spatial

            layer_dims[layer_path.name] = {
                "n_timesteps": n_timesteps,
                "n_spatial": n_spatial,
                "n_features": n_features,
                "samples_per_prompt": samples_per_prompt,
            }

            print(
                f"  {layer_path.name}: [{n_timesteps} ts × {n_spatial} sp × {n_features} feat] "
                f"= {samples_per_prompt:,} samples/prompt"
            )

    return layer_dims


def calculate_total_samples(
    num_prompts: int,
    layer_dims: Dict[str, Dict[str, int]],
    safety_margin: float = 1.0,
) -> Dict[str, int]:
    """
    Calculate total samples needed for pre-allocation.

    Args:
        num_prompts: Total number of prompts (across ALL array jobs)
        layer_dims: Output from calculate_layer_dimensions()
        safety_margin: Extra buffer (1.0 = exact, no margin needed)

    Returns:
        Dict mapping layer_name to total samples needed
    """
    totals = {}
    for layer_name, dims in layer_dims.items():
        samples = dims["samples_per_prompt"] * num_prompts
        total_with_margin = int(math.ceil(samples * safety_margin))
        totals[layer_name] = total_with_margin

    print(f"📊 Total samples needed for {num_prompts:,} prompts:")
    for layer_name, total in totals.items():
        print(f"  {layer_name}: {total:,}")

    return totals


def initialize_cache_from_dimensions(
    cache: "RepresentationCache",
    layer_dims: Dict[str, Dict[str, int]],
    total_samples: Dict[str, int],
    total_prompts: int,
) -> None:
    """
    Pre-initialize all layer memmaps with exact sizes.

    Args:
        cache: RepresentationCache instance
        layer_dims: Output from calculate_layer_dimensions()
        total_samples: Output from calculate_total_samples()
        total_prompts: Total number of prompts across all jobs
    """
    print("📦 Pre-initializing cache layers...")

    for layer_name, dims in layer_dims.items():
        if layer_name in total_samples:
            cache.initialize_layer(
                layer_name=layer_name,
                total_samples=total_samples[layer_name],
                feature_dim=dims["n_features"],
                total_prompts=total_prompts,
                samples_per_prompt=dims["samples_per_prompt"],
            )

    print("✅ All layers pre-initialized!")
