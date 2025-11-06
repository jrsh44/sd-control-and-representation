#!/usr/bin/env python3
"""
Efficient cache management for representation storage.
Provides efficient batched saving using Parquet format and memory-mapped loading.
Supports FP16 compression and append-only writes for scalability.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset


class RepresentationCache:
    """
    Manages representation storage using Parquet format with HuggingFace Datasets.

    Storage Format: Parquet shards for append-only writes
    Structure: {cache_dir}/{layer_name}/data-*.parquet
    Each dataset contains: object, style, prompt_nr, prompt_text, representation

    Features:
    - FP16 compression (50% size reduction)
    - Append-only Parquet writes (constant memory usage)
    - Backward compatible with legacy Arrow format
    - Flattened tensors with original_shape for large tensor support
    """

    def __init__(self, cache_dir: Path, use_fp16: bool = True):
        """
        Initialize cache manager.

        Args:
            cache_dir: Base directory for cache (e.g., results/sd_1_5_arrow)
            use_fp16: If True, store representations as float16 (50% size reduction)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_fp16 = use_fp16

        # Cache loaded datasets in memory
        self._loaded_datasets = {}

        # Cache existence indices for fast lookups
        self._existence_indices = {}

    def get_dataset_path(self, layer_name: str) -> Path:
        """Get path for layer dataset."""
        return self.cache_dir / f"{layer_name.lower()}"

    def load_layer_dataset(self, layer_name: str, use_cache: bool = True) -> Optional[Dataset]:
        """
        Load existing dataset for a layer from Parquet shards.
        Reads all data-*.parquet files and combines them into a single Dataset.

        Args:
            layer_name: Name of the layer
            use_cache: If True, cache loaded dataset in memory

        Returns:
            Dataset or None if doesn't exist
        """
        # Check in-memory cache first
        if use_cache and layer_name in self._loaded_datasets:
            return self._loaded_datasets[layer_name]

        dataset_path = self.get_dataset_path(layer_name)
        if dataset_path.exists():
            try:
                # Find all Parquet shard files
                shard_files = sorted(dataset_path.glob("data-*.parquet"))

                if not shard_files:
                    # Try legacy Arrow format for backward compatibility
                    arrow_files = list(dataset_path.glob("data-*.arrow"))
                    if arrow_files:
                        # Load with memory mapping (legacy format)
                        dataset = Dataset.load_from_disk(str(dataset_path))
                        if use_cache:
                            self._loaded_datasets[layer_name] = dataset
                        return dataset
                    return None

                # Read all Parquet shards and concatenate
                tables = [pq.read_table(str(shard)) for shard in shard_files]
                combined_table = pa.concat_tables(tables)

                # Wrap in Dataset
                dataset = Dataset(combined_table)

                if use_cache:
                    self._loaded_datasets[layer_name] = dataset

                return dataset
            except Exception as e:
                print(f"Warning: Could not load dataset from {dataset_path}: {e}")

        return None

    # def _build_existence_index(self, layer_name: str) -> set:
    #     """
    #     Build an index of existing (object, style, prompt_nr) tuples for fast lookup.

    #     Args:
    #         layer_name: Name of the layer

    #     Returns:
    #         Set of (object, style, prompt_nr) tuples
    #     """
    #     if layer_name in self._existence_indices:
    #         return self._existence_indices[layer_name]

    #     dataset = self.load_layer_dataset(layer_name)
    #     if dataset is None:
    #         self._existence_indices[layer_name] = set()
    #         return self._existence_indices[layer_name]

    #     print(f"    🔍 Building existence index for {layer_name} ({len(dataset)} items)...")

    #     # Vectorized approach - much faster!
    #     objects = dataset["object"]
    #     styles = dataset["style"]
    #     prompt_nrs = dataset["prompt_nr"]

    #     index = set(zip(objects, styles, prompt_nrs, strict=True))

    #     self._existence_indices[layer_name] = index
    #     print(f"    ✅ Index built with {len(index)} entries")
    #     return index

    # def check_exists(self, layer_name: str, object_name: str, style: str, prompt_nr: int) -> bool:
    #     """
    #     Check if specific representation exists in dataset.

    #     Args:
    #         layer_name: Name of the layer
    #         object_name: Object name
    #         style: Style name
    #         prompt_nr: Prompt number

    #     Returns:
    #         True if exists, False otherwise
    #     """
    #     index = self._build_existence_index(layer_name)
    #     return (object_name, style, prompt_nr) in index

    # def check_batch_exists(
    #     self, layer_name: str, queries: List[Tuple[str, str, int]]
    # ) -> List[bool]:
    #     """
    #     Efficiently check if multiple representations exist.

    #     Args:
    #         layer_name: Name of the layer
    #         queries: List of (object_name, style, prompt_nr) tuples

    #     Returns:
    #         List of booleans indicating existence for each query
    #     """
    #     # Use the same index as check_exists for consistency and speed
    #     index = self._build_existence_index(layer_name)
    #     return [query in index for query in queries]

    def save_batch(self, layer_name: str, batch_data: List[Dict], verbose: bool = True):
        """
        Save a batch of representations for a layer using append-only Parquet writes.
        Each batch is written as a separate Parquet shard without loading existing data.

        Args:
            layer_name: Name of the layer
            batch_data: List of dicts with keys:
                - object: str
                - style: str
                - prompt_nr: int
                - prompt_text: str
                - representation: torch.Tensor
            verbose: If True, print save information
        """
        if not batch_data:
            return

        dataset_path = self.get_dataset_path(layer_name)

        try:
            # Create directory if it doesn't exist
            dataset_path.mkdir(parents=True, exist_ok=True)

            # Convert batch to flattened numpy arrays
            flat_representations = []
            original_shapes = []

            for r in batch_data:
                tensor = (
                    r["representation"].cpu().half() if self.use_fp16 else r["representation"].cpu()
                )
                original_shapes.append(list(tensor.shape))
                flat_representations.append(tensor.flatten().numpy())

            # Create PyArrow table with large_list for representations
            pa_schema = pa.schema(
                [
                    ("object", pa.string()),
                    ("style", pa.string()),
                    ("prompt_nr", pa.int64()),
                    ("prompt_text", pa.string()),
                    (
                        "representation",
                        pa.large_list(pa.float16() if self.use_fp16 else pa.float32()),
                    ),
                    ("original_shape", pa.large_list(pa.int64())),
                ]
            )

            pa_table = pa.table(
                {
                    "object": [r["object"] for r in batch_data],
                    "style": [r["style"] for r in batch_data],
                    "prompt_nr": [r["prompt_nr"] for r in batch_data],
                    "prompt_text": [r["prompt_text"] for r in batch_data],
                    "representation": flat_representations,
                    "original_shape": original_shapes,
                },
                schema=pa_schema,
            )

            # Find next shard number by counting existing parquet files
            existing_shards = sorted(dataset_path.glob("data-*.parquet"))
            next_shard = len(existing_shards)

            # Write as new Parquet shard (append-only, no loading existing data)
            shard_path = dataset_path / f"data-{next_shard:05d}.parquet"
            pq.write_table(pa_table, shard_path, compression="snappy")

            if verbose:
                # Calculate size of this batch
                batch_size_mb = shard_path.stat().st_size / (1024**2)

                # Calculate total size and count
                total_size_mb = sum(
                    f.stat().st_size for f in dataset_path.glob("data-*.parquet")
                ) / (1024**2)
                total_shards = next_shard + 1

                dtype_str = "FP16" if self.use_fp16 else "FP32"

                if next_shard == 0:
                    print(
                        f"    💾 Created {layer_name} with {len(batch_data)} items "
                        f"({batch_size_mb:.1f} MB, shard #1, {dtype_str})"
                    )
                else:
                    print(
                        f"    💾 Appended {len(batch_data)} to {layer_name} "
                        f"(shard #{total_shards}, batch: {batch_size_mb:.1f} MB, "
                        f"total: {total_size_mb:.1f} MB, {dtype_str})"
                    )

            # Clear memory immediately
            del pa_table
            del flat_representations

        except Exception as e:
            raise e
        finally:
            # Clear cache for this layer to free memory
            if layer_name in self._loaded_datasets:
                del self._loaded_datasets[layer_name]
            if layer_name in self._existence_indices:
                del self._existence_indices[layer_name]

    def get_representation(
        self, layer_name: str, object_name: str, style: str, prompt_nr: int
    ) -> Optional[torch.Tensor]:
        """
        Load a single representation.

        Args:
            layer_name: Name of the layer
            object_name: Object name
            style: Style name
            prompt_nr: Prompt number

        Returns:
            Tensor (automatically converted to FP32 and reshaped) or None if not found
        """
        dataset = self.load_layer_dataset(layer_name)
        if dataset is None:
            return None

        matches = dataset.filter(
            lambda x: (
                x["object"] == object_name and x["style"] == style and x["prompt_nr"] == prompt_nr
            )
        )

        if len(matches) > 0:
            tensor = torch.from_numpy(matches[0]["representation"])
            # Convert back to FP32 if stored as FP16
            if tensor.dtype == torch.float16:
                tensor = tensor.float()
            # Reshape back to original shape if stored
            if "original_shape" in matches[0]:
                tensor = tensor.reshape(matches[0]["original_shape"])
            return tensor

        return None
