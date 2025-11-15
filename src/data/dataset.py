"""
PyTorch Dataset wrapper for cached representations.
Works with Arrow storage format from cache.py (HuggingFace datasets compatible).
"""

import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


# Thread-local storage for timing statistics
_thread_local = threading.local()


def _get_timing_stats():
    """Get or create timing stats for current thread."""
    if not hasattr(_thread_local, "timing_stats"):
        _thread_local.timing_stats = {
            "total_time": 0.0,
            "record_time": 0.0,
            "extract_time": 0.0,
            "transform_time": 0.0,
            "count": 0,
        }
    return _thread_local.timing_stats


def _reset_timing_stats():
    """Reset timing stats for current thread."""
    if hasattr(_thread_local, "timing_stats"):
        _thread_local.timing_stats = {
            "total_time": 0.0,
            "record_time": 0.0,
            "extract_time": 0.0,
            "transform_time": 0.0,
            "count": 0,
        }


class RepresentationDataset(Dataset):
    """
    PyTorch Dataset for cached representations with lazy loading.
    Loads from HuggingFace datasets format created by cache.py.
    """

    def __init__(
        self,
        dataset_path: Path,
        flatten: bool = True,
        filter_fn: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            dataset_path: Path to layer directory
                (e.g., results/model/cached_representations/unet_mid_att)
            flatten: If True, flatten representations to 1D vectors
            filter_fn: Optional function to filter dataset
                (e.g., lambda x: x['object'] == 'cat' and x['style'] == 'Impressionism')
            transform: Optional transform function applied to representations
            return_metadata: If True, return (representation, metadata) tuple
        """
        self.dataset_path = Path(dataset_path)
        self.flatten = flatten
        self.transform = transform
        self.return_metadata = return_metadata

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset not found: {dataset_path}")

        # Load HuggingFace dataset
        print(f"Loading dataset from {dataset_path}...")
        load_start = time.time()
        self.hf_dataset = load_from_disk(str(dataset_path))
        load_time = time.time() - load_start
        print(f"Loaded {len(self.hf_dataset)} records in {load_time:.2f}s")

        # Apply filtering if needed
        if filter_fn is not None:
            print(f"Filtering dataset... (original size: {len(self.hf_dataset)})")
            filter_start = time.time()
            self.hf_dataset = self.hf_dataset.filter(filter_fn)
            filter_time = time.time() - filter_start
            print(f"Filtered dataset size: {len(self.hf_dataset)} in {filter_time:.2f}s")

        # Get feature dimension from first sample
        if len(self.hf_dataset) > 0:
            first_sample = self.hf_dataset[0]
            self.feature_dim = len(first_sample["list_of_features"])
            print(f"Feature dimension: {self.feature_dim}")
        else:
            raise ValueError("Dataset is empty")

        self.hf_dataset.with_format("torch")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        """
        Get item at index.

        Returns:
            If return_metadata=True: (representation, metadata) tuple
            If return_metadata=False: representation tensor only
        """
        item_start = time.time()
        stats = _get_timing_stats()

        # Get record from HuggingFace dataset
        record_start = time.time()
        record = self.hf_dataset[idx]
        record_time = time.time() - record_start

        # Extract features
        extract_start = time.time()
        features = record["list_of_features"]
        rep = torch.tensor(features, dtype=torch.float32)
        extract_time = time.time() - extract_start

        # Apply transform
        transform_time = 0.0
        if self.transform is not None:
            transform_start = time.time()
            rep = self.transform(rep)
            transform_time = time.time() - transform_start

        total_time = time.time() - item_start

        # Accumulate stats
        stats["total_time"] += total_time
        stats["record_time"] += record_time
        stats["extract_time"] += extract_time
        stats["transform_time"] += transform_time
        stats["count"] += 1

        # Log detailed timing for first 10 samples
        if idx < 10:
            print(
                f"[Dataset __getitem__ idx={idx}] "
                f"Total: {total_time * 1000:.2f}ms "
                f"(record: {record_time * 1000:.2f}ms, "
                f"extract: {extract_time * 1000:.2f}ms, "
                f"transform: {transform_time * 1000:.2f}ms)"
            )

        if not self.return_metadata:
            return rep

        # Return representation and metadata
        metadata = {
            "timestep": record["timestep"],
            "spatial": record["spatial"],
            "object": record["object"],
            "style": record["style"],
            "prompt_nr": record["prompt_nr"],
            "prompt_text": record["prompt_text"],
            "num_steps": record["num_steps"],
            "guidance_scale": record["guidance_scale"],
        }

        return (rep, metadata)

    @staticmethod
    def get_timing_stats():
        """Get accumulated timing statistics for current thread."""
        return _get_timing_stats()

    @staticmethod
    def reset_timing_stats():
        """Reset accumulated timing statistics for current thread."""
        _reset_timing_stats()

    @staticmethod
    def get_available_values(dataset_path: Path, column: str) -> List[str]:
        """
        Get all unique values for a metadata column.

        Args:
            dataset_path: Path to layer directory
            column: Metadata column name ('style', 'object', etc.)

        Returns:
            List of unique values sorted alphabetically
        """
        ds = load_from_disk(str(dataset_path))
        if column not in ds.column_names:
            raise ValueError(f"Column '{column}' not found. Available: {ds.column_names}")

        unique_values = set(ds[column])
        # Type ignore for pyright - sorted() can handle sets of strings
        return sorted(unique_values)  # type: ignore

    @staticmethod
    def get_metadata_summary(dataset_path: Path) -> pd.DataFrame:
        """
        Get summary statistics for all metadata columns.

        Args:
            dataset_path: Path to layer directory

        Returns:
            DataFrame with value counts for each column
        """
        ds = load_from_disk(str(dataset_path))

        summary = {}
        metadata_cols = [
            "object",
            "style",
            "timestep",
            "spatial",
            "prompt_nr",
            "num_steps",
            "guidance_scale",
        ]
        for col in metadata_cols:
            if col in ds.column_names:
                # Get column values as a list
                values = ds[col]
                # Type ignore for pyright - pd.Series can handle dataset columns
                value_counts = pd.Series(list(values)).value_counts()  # type: ignore
                summary[col] = {
                    "unique_values": len(value_counts),
                    "top_5": value_counts.head(5).to_dict(),
                    "total_count": len(ds),
                }

        return pd.DataFrame(summary).T
