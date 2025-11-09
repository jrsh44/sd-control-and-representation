"""
PyTorch Dataset wrapper for cached representations.
Works with Parquet storage format from cache.py.
"""

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class RepresentationDataset(Dataset):
    """
    PyTorch Dataset for cached representations with lazy loading.
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
            filter_fn: Optional function to filter dataset (e.g., lambda x: x['object'] == 'cat')
            transform: Optional transform function applied to representations
            return_metadata: If True, return (representation, metadata) tuple
        """
        self.dataset_path = Path(dataset_path)
        self.flatten = flatten
        self.transform = transform
        self.return_metadata = return_metadata

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset not found: {dataset_path}")

        # Load metadata
        metadata_path = self.dataset_path / "metadata.parquet"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")

        self.metadata_table = pq.read_table(str(metadata_path))
        self.metadata_df = self.metadata_table.to_pandas()

        # Store file paths only
        self.data_files = sorted(self.dataset_path.glob("data-*.parquet"))
        if not self.data_files:
            raise ValueError(f"No data files found in {dataset_path}")

        # Verify counts match
        if len(self.data_files) != len(self.metadata_df):
            raise ValueError(
                f"Mismatch: {len(self.metadata_df)} metadata entries but "
                f"{len(self.data_files)} data files"
            )

        # Apply filtering if needed
        if filter_fn is not None:
            print(f"Filtering dataset... (original size: {len(self.metadata_df)})")
            mask = self.metadata_df.apply(lambda row: filter_fn(row.to_dict()), axis=1)
            # Store original indices to map filtered idx -> file idx
            self.file_indices = self.metadata_df[mask].index.tolist()
            self.metadata_df = self.metadata_df[mask].reset_index(drop=True)
            print(f"Filtered dataset size: {len(self.metadata_df)}")
        else:
            # No filtering: direct 1-to-1 mapping
            self.file_indices = list(range(len(self.metadata_df)))

        # Get shape info from first file only
        if self.data_files:
            first_table = pq.read_table(str(self.data_files[0]), memory_map=True)
            first_shape = first_table["original_shape"][0].as_py()
            self.original_shape = tuple(first_shape)

            if flatten:
                self.feature_dim = int(np.prod(self.original_shape))
            else:
                self.feature_dim = self.original_shape

            print(
                f"Dataset initialized: {len(self.metadata_df)} samples, "
                f"original shape: {self.original_shape}, "
                f"feature_dim: {self.feature_dim}"
            )
        else:
            raise ValueError("No data files found")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        """
        Get item at index.

        Returns:
            If return_metadata=True: (representation, metadata) tuple
            If return_metadata=False: representation tensor only
        """
        # Get metadata
        metadata_row = self.metadata_df.iloc[idx]

        # Map filtered index to original file index
        file_idx = self.file_indices[idx]

        # Load ONLY this specific file (lazy, memory-mapped)
        file_path = self.data_files[file_idx]
        tensor_table = pq.read_table(str(file_path), memory_map=True)

        # Extract representation
        rep_array = tensor_table["representation"][0].as_py()
        rep = torch.tensor(rep_array, dtype=torch.float32)

        # Reshape if needed
        if not self.flatten:
            shape_array = tensor_table["original_shape"][0].as_py()
            rep = rep.reshape(shape_array)

        # Apply transform
        if self.transform is not None:
            rep = self.transform(rep)

        if not self.return_metadata:
            return rep

        # Return representation and metadata
        metadata = {
            "object": metadata_row["object"],
            "style": metadata_row["style"],
            "prompt_nr": metadata_row["prompt_nr"],
            "prompt_text": metadata_row["prompt_text"],
        }

        return (rep, metadata)

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
        metadata_path = Path(dataset_path) / "metadata.parquet"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")

        metadata_df = pd.read_parquet(metadata_path)
        if column not in metadata_df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {list(metadata_df.columns)}")

        return sorted(metadata_df[column].dropna().unique().tolist())

    @staticmethod
    def get_metadata_summary(dataset_path: Path) -> pd.DataFrame:
        """
        Get summary statistics for all metadata columns.

        Args:
            dataset_path: Path to layer directory

        Returns:
            DataFrame with value counts for each column
        """
        metadata_path = Path(dataset_path) / "metadata.parquet"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")

        metadata_df = pd.read_parquet(metadata_path)

        summary = {}
        for col in metadata_df.columns:
            value_counts = metadata_df[col].value_counts()
            summary[col] = {
                "unique_values": len(value_counts),
                "top_5": value_counts.head(5).to_dict(),
                "total_count": len(metadata_df),
            }

        return pd.DataFrame(summary).T
