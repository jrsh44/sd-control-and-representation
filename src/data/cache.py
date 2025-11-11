#!/usr/bin/env python3
"""
Cache management for representation storage using Arrow format.
Compatible with HuggingFace datasets.load_from_disk().
"""

import fcntl
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value
from datasets.fingerprint import generate_fingerprint


class FileLock:
    """Simple file-based lock for process synchronization."""

    def __init__(self, lock_file: Path, timeout: int = 300):
        self.lock_file = lock_file
        self.timeout = timeout
        self.lock_fd = None

    def __enter__(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_fd = open(self.lock_file, "w")

        start_time = time.time()
        while True:
            try:
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except IOError as e:
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(
                        f"Could not acquire lock on {self.lock_file} after {self.timeout}s"
                    ) from e
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_fd:
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
            self.lock_fd.close()


class RepresentationCache:
    def __init__(self, cache_dir: Path, use_fp16: bool = True):
        """
        Args:
            cache_dir: Base directory for cache
            use_fp16: Store as float16 (50% size reduction)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_fp16 = use_fp16

        # Track metadata for quick existence checks
        # layer -> {(object, style, prompt_nr): shard_id}
        self._metadata_index: Dict[str, Dict] = {}

        # Lock directory for synchronization
        self.lock_dir = self.cache_dir / ".locks"
        self.lock_dir.mkdir(exist_ok=True)

    def get_dataset_path(self, layer_name: str) -> Path:
        """
        Get storage path for a layer's dataset.

        Args:
            layer_name: Name of the layer

        Returns:
            Path: Directory path for the layer
        """
        return self.cache_dir / f"{layer_name.lower()}"

    def _get_lock_file(self, layer_name: str) -> Path:
        """Get lock file path for a specific layer."""
        return self.lock_dir / f"{layer_name.lower()}.lock"

    def _update_metadata_on_disk(self, layer_name: str):
        """
        Update metadata index in dataset_info.json atomically.
        Must be called within a lock context.
        """
        dataset_path = self.get_dataset_path(layer_name)
        dataset_info_path = dataset_path / "dataset_info.json"

        # Load existing info or create new
        if dataset_info_path.exists():
            with open(dataset_info_path, "r") as f:
                info = json.load(f)
        else:
            info = {}

        # Convert metadata index to JSON-serializable format
        metadata_index = {
            f"{k[0]}|{k[1]}|{k[2]}": shard_id
            for k, shard_id in self._metadata_index[layer_name].items()
        }
        info["metadata_index"] = metadata_index

        # Atomic write: write to temp file, then rename
        temp_path = dataset_info_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(info, f, indent=2)
        temp_path.replace(dataset_info_path)

    def _load_metadata_index(self, layer_name: str):
        """Load metadata index from dataset_info.json if exists."""
        if layer_name in self._metadata_index:
            return

        dataset_path = self.get_dataset_path(layer_name)
        dataset_info_path = dataset_path / "dataset_info.json"

        self._metadata_index[layer_name] = {}

        if dataset_info_path.exists():
            try:
                with open(dataset_info_path, "r") as f:
                    info = json.load(f)

                    # Load metadata index if exists
                    if "metadata_index" in info:
                        for key_str, shard_id in info["metadata_index"].items():
                            # Parse key: "object|style|prompt_nr"
                            parts = key_str.split("|")
                            if len(parts) == 3:
                                key = (parts[0], parts[1], int(parts[2]))
                                self._metadata_index[layer_name][key] = shard_id
            except Exception as e:
                print(f"Warning: Could not load metadata index: {e}")

    def check_exists(self, layer_name: str, object_name: str, style: str, prompt_nr: int) -> bool:
        """
        Check if a specific representation exists (thread-safe).

        Args:
            layer_name: Name of the layer
            object_name: Object name
            style: Style name
            prompt_nr: Prompt number

        Returns:
            bool: True if representation exists
        """
        lock_file = self._get_lock_file(layer_name)
        with FileLock(lock_file, timeout=30):
            self._load_metadata_index(layer_name)
            key = (object_name, style, prompt_nr)
            return key in self._metadata_index.get(layer_name, {})

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
        Save a single representation to disk (thread-safe).
        Uses unique UUIDs for shard names to avoid conflicts.

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
        dataset_path = self.get_dataset_path(layer_name)
        tmp_shards_dir = dataset_path / ".tmp_shards"
        tmp_shards_dir.mkdir(parents=True, exist_ok=True)

        # Convert tensor to CPU and optionally to fp16
        tensor = representation.cpu().half() if self.use_fp16 else representation.cpu()

        # Expect shape: [timesteps, batch, spatial, features]
        if tensor.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [timesteps, batch, spatial, features], "
                f"got shape: {tensor.shape}"
            )

        # Squeeze batch dimension
        tensor = tensor.squeeze(1)  # [timesteps, spatial, features]
        n_timesteps, n_spatial, n_features = tensor.shape

        # OPTIMIZED: Create arrays directly instead of list of dicts
        total_records = n_timesteps * n_spatial

        # Pre-allocate arrays
        timesteps_array = np.repeat(np.arange(n_timesteps, dtype=np.int32), n_spatial)
        spatial_array = np.tile(np.arange(n_spatial, dtype=np.int32), n_timesteps)

        # Reshape features: [timesteps, spatial, features] -> [total_records, features]
        features_array = tensor.reshape(total_records, n_features).numpy()

        # Create constant arrays efficiently
        features_type = "float16" if self.use_fp16 else "float32"

        ds = Dataset.from_dict(
            {
                "timestep": timesteps_array,
                "spatial": spatial_array,
                "list_of_features": features_array.tolist(),
                "object": [object_name] * total_records,
                "style": [style] * total_records,
                "guidance_scale": [guidance_scale] * total_records,
                "prompt_nr": [prompt_nr] * total_records,
                "prompt_text": [prompt_text] * total_records,
                "num_steps": [num_steps] * total_records,
            },
            features=Features(
                {
                    "timestep": Value("int32"),
                    "spatial": Value("int32"),
                    "list_of_features": Sequence(Value(features_type)),
                    "object": Value("string"),
                    "style": Value("string"),
                    "guidance_scale": Value("float32"),
                    "prompt_nr": Value("int32"),
                    "prompt_text": Value("string"),
                    "num_steps": Value("int32"),
                }
            ),
        )

        # THREAD-SAFE: Use UUID for unique shard name instead of counter
        # This avoids race conditions when multiple processes write simultaneously
        shard_id = str(uuid.uuid4())
        shard_path = tmp_shards_dir / f"shard_{shard_id}"

        # Save the shard
        ds.save_to_disk(str(shard_path), num_proc=1)

        # Update metadata index atomically
        lock_file = self._get_lock_file(layer_name)
        with FileLock(lock_file, timeout=60):
            # Reload index to get latest state
            self._load_metadata_index(layer_name)

            # Update with new entry
            key = (object_name, style, prompt_nr)
            self._metadata_index[layer_name][key] = shard_id

            # Persist to disk immediately for other processes
            self._update_metadata_on_disk(layer_name)

    @staticmethod
    def _consolidate_shards(tmp_shards_dir: Path, output_dir: Path) -> Dataset:
        """
        Consolidate sharded datasets into a single directory.

        Args:
            tmp_shards_dir: Directory containing shard_* subdirectories (UUID-named)
            output_dir: Final output directory

        Returns:
            Consolidated Dataset
        """
        # Get all shards (now with UUID names)
        shard_dirs = sorted(tmp_shards_dir.glob("shard_*"))

        if not shard_dirs:
            raise ValueError(f"No shards found in {tmp_shards_dir}")

        # Copy dataset_info.json from first shard
        shutil.copy2(
            shard_dirs[0] / "dataset_info.json",
            output_dir / "dataset_info.json",
        )

        arrow_files = []
        file_count = 0

        # Move all files from shards to output directory
        for shard_dir in shard_dirs:
            # state.json contains filenames
            state_path = shard_dir / "state.json"
            if not state_path.exists():
                print(f"Warning: {shard_dir} missing state.json, skipping...")
                continue

            state = json.loads(state_path.read_text())

            for data_file in state["_data_files"]:
                original_name = data_file["filename"]
                src_path = shard_dir / original_name

                if not src_path.exists():
                    print(f"Warning: {src_path} not found, skipping...")
                    continue

                # Rename to sequential numbering
                new_name = f"data-{file_count:05d}-of-XXXXX.arrow"
                dst_path = output_dir / new_name

                shutil.move(str(src_path), str(dst_path))
                arrow_files.append({"filename": new_name})
                file_count += 1

        # Update total count in filenames
        total_files = len(arrow_files)
        for i, file_info in enumerate(arrow_files):
            old_name = file_info["filename"]
            new_name = old_name.replace("XXXXX", f"{total_files:05d}")
            old_path = output_dir / old_name
            new_path = output_dir / new_name
            shutil.move(str(old_path), str(new_path))
            arrow_files[i]["filename"] = new_name

        # Create state.json with proper structure
        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }

        with open(output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        # Load dataset to generate fingerprint
        ds = Dataset.load_from_disk(str(output_dir))
        fingerprint = generate_fingerprint(ds)

        # Update state.json with fingerprint
        with open(output_dir / "state.json", "r+") as f:
            state = json.load(f)
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

        # Clean up temporary shards directory
        shutil.rmtree(tmp_shards_dir)

        return ds

    def save_metadata(self):
        """
        Consolidate all sharded files into proper HuggingFace dataset structure.
        Thread-safe: only one process can consolidate at a time per layer.
        """
        for layer_name in list(self._metadata_index.keys()):
            lock_file = self._get_lock_file(layer_name)

            # Try to acquire lock, skip if another process is consolidating
            try:
                with FileLock(lock_file, timeout=5):
                    dataset_path = self.get_dataset_path(layer_name)
                    tmp_shards_dir = dataset_path / ".tmp_shards"

                    if not tmp_shards_dir.exists():
                        print(f"No shards found for layer {layer_name}, skipping...")
                        continue

                    shard_dirs = sorted(tmp_shards_dir.glob("shard_*"))
                    if not shard_dirs:
                        print(f"No shard directories found for layer {layer_name}, skipping...")
                        continue

                    print(f"Consolidating {len(shard_dirs)} shards for layer {layer_name}...")

                    # Consolidate without loading data
                    ds = self._consolidate_shards(tmp_shards_dir, dataset_path)

                    # Metadata index already updated in save_representation
                    print(f"✅ Saved consolidated dataset for {layer_name}: {len(ds)} records")
            except TimeoutError:
                print(f"⏭️  Skipping {layer_name} consolidation (another process is working on it)")
                continue
