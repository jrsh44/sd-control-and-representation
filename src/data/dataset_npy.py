"""
Fast memmap-based dataset for representations.
200x faster than Arrow format for random access patterns.
"""

import json
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NPYDataset(Dataset):
    """
    Fast memmap-based dataset with filtering support.
    Uses lightweight index file for fast filtering without loading full metadata.
    Compatible with DataLoader shuffling.
    """

    def __init__(
        self,
        cache_dir: Path,
        layer_name: str,
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
        filter_fn: Optional[Callable] = None,
        indices: Optional[List[int]] = None,
    ):
        """
        Args:
            cache_dir: Base cache directory
            layer_name: Layer name (e.g., 'unet_up_1_att_1')
            transform: Optional transform function
            return_metadata: If True, return (data, metadata) tuple
            filter_fn: Optional filter function: (index_entry) -> bool
                       Works on lightweight index entries (no prompt_text)
            indices: Pre-computed indices to use (overrides filter_fn)
        """
        layer_dir = cache_dir / layer_name

        if not layer_dir.exists():
            raise ValueError(f"Layer not found: {layer_dir}")

        # Load as memmap - INSTANT, no RAM used!
        # Auto-detect format: pure memmap (new) vs np.save with pickle (old)
        data_path = layer_dir / "data.npy"
        try:
            # Try pure memmap first (new format, fastest)
            self._full_data = np.load(str(data_path), mmap_mode="r", allow_pickle=False)
        except Exception as e:
            # Could be ValueError, OSError, or _pickle.UnpicklingError
            # Fallback to old format with pickle
            try:
                self._full_data = np.load(str(data_path), mmap_mode="r", allow_pickle=True)
                print("  ⚠️  Using old format (np.save with pickle)")
                print("  💡 Tip: Regenerate cache for faster loading")
            except Exception as fallback_error:
                raise ValueError(
                    f"Failed to load {data_path}. "
                    f"Error without pickle: {e}. "
                    f"Error with pickle: {fallback_error}"
                ) from None

        # Load lightweight index (for filtering)
        index_path = layer_dir / "index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                self._index = json.load(f)
        else:
            # Fallback: create index from metadata
            import pickle

            meta_path = layer_dir / "metadata.pkl"
            with open(meta_path, "rb") as f:
                full_metadata = pickle.load(f)

            self._index = [
                {
                    "timestep": m["timestep"],
                    "spatial": m["spatial"],
                    "object": m["object"],
                    "style": m["style"],
                    "prompt_nr": m["prompt_nr"],
                    "num_steps": m["num_steps"],
                    "guidance_scale": m["guidance_scale"],
                }
                for m in full_metadata
            ]

        # Load full metadata only if needed
        self._full_metadata = None
        self.return_metadata = return_metadata
        if return_metadata:
            meta_path = layer_dir / "metadata.pkl"
            with open(meta_path, "rb") as f:
                self._full_metadata = pickle.load(f)

        # Apply filtering
        if indices is not None:
            # Use pre-computed indices
            self.indices = indices
            print(f"  Using {len(indices)} pre-filtered indices")
        elif filter_fn is not None:
            # Apply filter on lightweight index
            print("  Applying filter function on index...")
            self.indices = [i for i, entry in enumerate(self._index) if filter_fn(entry)]
            print(f"  Filtered: {len(self.indices)}/{len(self._index)} samples")
        else:
            # No filtering - use all data
            self.indices = list(range(len(self._full_data)))

        self.transform = transform

        print(f"Loaded NPY dataset from {data_path}")
        print(f"  Total samples: {len(self._full_data)}")
        print(f"  Active samples: {len(self)} (after filtering)")
        print(f"  Data shape: {self._full_data.shape}")
        print(f"  Dtype: {self._full_data.dtype}")
        print(f"  Size on disk: {data_path.stat().st_size / 1e9:.2f} GB")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map filtered index to original index
        real_idx = self.indices[idx]

        # Direct memmap access - FAST! (~0.05ms)
        # Automatically converts fp16 → fp32 for training
        rep = torch.from_numpy(self._full_data[real_idx]).float()

        if self.transform is not None:
            rep = self.transform(rep)

        if not self.return_metadata:
            return rep

        # Return with metadata (if loaded)
        if self._full_metadata is not None:
            return rep, self._full_metadata[real_idx]
        else:
            # Return lightweight index entry
            return rep, self._index[real_idx]

    def get_available_values(self, column: str) -> List[str]:
        """
        Get all unique values for a metadata column.

        Args:
            column: Column name ('style', 'object', 'timestep', etc.)

        Returns:
            List of unique values sorted
        """
        if column not in self._index[0]:
            raise ValueError(
                f"Column '{column}' not found. Available: {list(self._index[0].keys())}"
            )

        unique_values = {entry[column] for entry in self._index}
        return sorted(unique_values)

    def get_metadata_summary(self) -> dict:
        """
        Get summary statistics for metadata columns.

        Returns:
            Dict with stats for each column
        """
        from collections import Counter

        summary = {}
        for key in self._index[0].keys():
            values = [entry[key] for entry in self._index]
            counter = Counter(values)

            summary[key] = {
                "unique_values": len(counter),
                "top_5": dict(counter.most_common(5)),
                "total_count": len(values),
            }

        return summary
