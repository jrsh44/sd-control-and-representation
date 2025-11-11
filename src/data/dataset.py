"""
PyTorch Dataset wrapper for cached representations.
Works with Arrow storage format from cache.py (HuggingFace datasets compatible).
"""

from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


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
        self.hf_dataset = load_from_disk(str(dataset_path))
        print(f"Loaded {len(self.hf_dataset)} records")

        # Apply filtering if needed
        if filter_fn is not None:
            print(f"Filtering dataset... (original size: {len(self.hf_dataset)})")
            self.hf_dataset = self.hf_dataset.filter(filter_fn)
            print(f"Filtered dataset size: {len(self.hf_dataset)}")

        # Get feature dimension from first sample
        if len(self.hf_dataset) > 0:
            first_sample = self.hf_dataset[0]
            self.feature_dim = len(first_sample["list_of_features"])
            print(f"Feature dimension: {self.feature_dim}")
        else:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        """
        Get item at index.

        Returns:
            If return_metadata=True: (representation, metadata) tuple
            If return_metadata=False: representation tensor only
        """
        # Get record from HuggingFace dataset
        record = self.hf_dataset[idx]

        # Extract features
        features = record["list_of_features"]
        rep = torch.tensor(features, dtype=torch.float32)

        # Apply transform
        if self.transform is not None:
            rep = self.transform(rep)

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
