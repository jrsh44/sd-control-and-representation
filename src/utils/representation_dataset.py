"""
Efficient PyTorch Dataset wrappers for cached representations.
Supports memory mapping, streaming, and filtering for SAE training.
Works with both Parquet and legacy Arrow storage formats.
"""

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, IterableDataset


class RepresentationDataset(Dataset):
    """
    PyTorch Dataset for cached representations (map-style).

    Use Case:
        - Standard SAE training on a single layer/file at a time
        - When you need random access to samples and proper shuffling
        - Should work well with files up to 100-200GB in size

    Benefits:
        - True random shuffling across entire dataset
        - Multi-worker support for faster data loading
        - Random access by index for debugging/inspection
        - Works with all standard PyTorch tools
        - Memory efficient via memory mapping
        - Can create subsets with get_subset() method

    Recommended for: Most SAE training scenarios with large datasets
    """

    def __init__(
        self,
        dataset_path: Path,
        flatten: bool = True,
        filter_fn: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        return_metadata: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            dataset_path: Path to dataset directory (e.g., results/sd_1_5/unet_mid_att)
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

        # Load dataset with memory mapping
        self.dataset = HFDataset.load_from_disk(str(dataset_path))

        # Apply filtering if needed
        if filter_fn is not None:
            print(f"Filtering dataset... (original size: {len(self.dataset)})")
            self.dataset = self.dataset.filter(filter_fn)
            print(f"Filtered dataset size: {len(self.dataset)}")

        # Get shape info from first item
        if len(self.dataset) > 0:
            sample = self.dataset[0]["representation"]
            # Data is stored flattened, get original shape if available
            if "original_shape" in self.dataset[0]:
                self.original_shape = tuple(self.dataset[0]["original_shape"])
                if flatten:
                    self.feature_dim = int(np.prod(self.original_shape))
                else:
                    self.feature_dim = self.original_shape
            else:
                # Legacy format: data already in original shape
                self.original_shape = sample.shape
                if flatten:
                    self.feature_dim = int(np.prod(sample.shape))
                else:
                    self.feature_dim = sample.shape

            print(
                f"Dataset initialized: {len(self.dataset)} samples, "
                f"original shape: {self.original_shape}, "
                f"feature_dim: {self.feature_dim}"
            )
        else:
            raise ValueError("Dataset is empty after filtering")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get item at index.

        Returns:
            If return_metadata=True: (representation, metadata) tuple
            If return_metadata=False: representation tensor only
        """
        item = self.dataset[idx]
        item = dict(item)

        # Convert numpy array to torch tensor and always convert to FP32
        rep = torch.from_numpy(item["representation"]).float()

        # Reshape to original shape if needed (data is stored flattened)
        if "original_shape" in item and not self.flatten:
            rep = rep.reshape(item["original_shape"])
        elif self.flatten and len(rep.shape) > 1:
            # Legacy format: flatten if needed
            rep = rep.flatten()

        # Apply transform
        if self.transform is not None:
            rep = self.transform(rep)

        if not self.return_metadata:
            return rep

        # Return representation and metadata
        metadata = {
            "object": item["object"],
            "style": item["style"],
            "prompt_nr": item["prompt_nr"],
            "prompt_text": item["prompt_text"],
        }

        return rep, metadata

    def get_subset(self, indices: List[int]) -> "RepresentationDataset":
        """Create a subset dataset with given indices."""
        subset_dataset = self.dataset.select(indices)
        new_dataset = RepresentationDataset.__new__(RepresentationDataset)
        new_dataset.dataset_path = self.dataset_path
        new_dataset.flatten = self.flatten
        new_dataset.transform = self.transform
        new_dataset.return_metadata = self.return_metadata
        new_dataset.dataset = subset_dataset
        new_dataset.original_shape = self.original_shape
        new_dataset.feature_dim = self.feature_dim
        return new_dataset


class StreamingRepresentationDataset(IterableDataset):
    """
    Streaming version for very large datasets (iterable-style).
    Only loads batches into memory as needed.

    Use Case:
        - Extremely large datasets that cause issues even with memory mapping
        - When sequential access is acceptable
        - Limited RAM environments

    Benefits:
        - Minimal memory footprint
        - Can handle arbitrarily large datasets
        - Built-in buffer-based shuffling

    Trade-offs:
        - Buffer-based shuffling only (not true random shuffle)
        - No random access by index
        - Limited multi-worker benefits
        - Cannot determine dataset length easily
        - Harder to create train/val splits

    Recommended for: Only use if RepresentationDataset causes OOM errors
    """

    def __init__(
        self,
        dataset_path: Path,
        flatten: bool = True,
        filter_fn: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        shuffle: bool = True,
        buffer_size: int = 10000,
        return_metadata: bool = True,
    ):
        """
        Initialize streaming dataset.

        Args:
            dataset_path: Path to dataset directory
            flatten: If True, flatten representations to 1D vectors
            filter_fn: Optional function to filter dataset
            transform: Optional transform function
            shuffle: If True, shuffle with buffer
            buffer_size: Size of shuffle buffer
            return_metadata: If True, return (representation, metadata) tuple
        """
        self.dataset_path = Path(dataset_path)
        self.flatten = flatten
        self.transform = transform
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.return_metadata = return_metadata

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset not found: {dataset_path}")

        # Load dataset
        self.dataset = HFDataset.load_from_disk(str(dataset_path))

        if filter_fn is not None:
            self.dataset = self.dataset.filter(filter_fn)

        # Get feature dimension
        if len(self.dataset) > 0:
            sample = self.dataset[0]["representation"]
            if flatten:
                self.feature_dim = int(np.prod(sample.shape))
            else:
                self.feature_dim = sample.shape

    def __iter__(self):
        """Iterate through dataset with optional shuffling."""
        # Create iterator
        if self.shuffle:
            # datasets.Dataset.shuffle does not accept a buffer_size parameter
            # call shuffle() without buffer_size (or use seed if needed).
            dataset_iter = self.dataset.shuffle()
        else:
            dataset_iter = self.dataset
        for item in dataset_iter:
            item = dict(item)
            rep = torch.from_numpy(item["representation"]).float()

            # Reshape to original shape if needed (data is stored flattened)
            if "original_shape" in item and not self.flatten:
                rep = rep.reshape(item["original_shape"])
            elif self.flatten and len(rep.shape) > 1:
                # Legacy format: flatten if needed
                rep = rep.flatten()

            if self.transform is not None:
                rep = self.transform(rep)

            if not self.return_metadata:
                yield rep
            else:
                metadata = {
                    "object": item["object"],
                    "style": item["style"],
                    "prompt_nr": item["prompt_nr"],
                    "prompt_text": item["prompt_text"],
                }
                yield rep, metadata


class MultiLayerRepresentationDataset(Dataset):
    """
    Dataset that loads representations from multiple layers simultaneously (map-style).

    Use Case:
        - Training joint SAEs across multiple layers
        - Multi-layer comparative analysis (same sample, different layers)
        - Multi-task SAE training with layer-specific objectives
        - Analyzing cross-layer representation patterns

    Benefits:
        - Synchronized access to same sample across all layers
        - Ensures consistent indexing and ordering across layers
        - Memory efficient via memory mapping (like RepresentationDataset)
        - Returns dict of {layer_name: representation} for easy multi-layer processing
        - All benefits of map-style dataset (shuffling, random access, etc.)

    Trade-offs:
        - Higher RAM usage when loading many layers (e.g., 10 layers: 20-40GB RAM)
        - All layer files must have same number of samples
        - Slower per-sample access due to loading from multiple files

    Recommended for: Joint multi-layer SAE training or cross-layer analysis
    """

    def __init__(
        self,
        cache_dir: Path,
        layer_names: List[str],
        flatten: bool = True,
        filter_fn: Optional[Callable] = None,
        return_metadata: bool = True,
    ):
        """
        Initialize multi-layer dataset.

        Args:
            cache_dir: Base directory containing layer datasets
            layer_names: List of layer names (e.g., ['unet_mid_att', 'unet_down_1_att_1'])
            flatten: If True, flatten each layer's representations
            filter_fn: Optional filter applied to all layers
            return_metadata: If True, return metadata dict
        """
        self.cache_dir = Path(cache_dir)
        self.layer_names = layer_names
        self.flatten = flatten
        self.return_metadata = return_metadata

        # Load all layer datasets
        self.layer_datasets = {}
        for layer_name in layer_names:
            layer_path = cache_dir / layer_name.lower()
            if not layer_path.exists():
                raise ValueError(f"Layer dataset not found: {layer_path}")

            dataset = HFDataset.load_from_disk(str(layer_path))

            if filter_fn is not None:
                dataset = dataset.filter(filter_fn)

            self.layer_datasets[layer_name] = dataset

        # Verify all datasets have same length
        lengths = [len(ds) for ds in self.layer_datasets.values()]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"Layer datasets have different lengths: {
                    dict(zip(layer_names, lengths, strict=False))
                }"
            )

        self.length = lengths[0]

        # Get feature dimensions for each layer
        self.feature_dims = {}
        for layer_name, dataset in self.layer_datasets.items():
            if len(dataset) > 0:
                sample_shape = dataset[0]["representation"].shape
                if flatten:
                    self.feature_dims[layer_name] = int(np.prod(sample_shape))
                else:
                    self.feature_dims[layer_name] = sample_shape

        print(f"Multi-layer dataset initialized: {self.length} samples")
        print(f"Layers: {layer_names}")
        print(f"Feature dimensions: {self.feature_dims}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Get item at index from all layers.

        Returns:
            If return_metadata=True: (representations_dict, metadata) tuple
            If return_metadata=False: representations_dict only
        """
        representations = {}
        metadata = None
        for layer_name, dataset in self.layer_datasets.items():
            item = dataset[idx]
            item = dict(item)
            rep = torch.from_numpy(item["representation"]).float()

            # Reshape to original shape if needed (data is stored flattened)
            if "original_shape" in item and not self.flatten:
                rep = rep.reshape(item["original_shape"])
            elif self.flatten and len(rep.shape) > 1:
                # Legacy format: flatten if needed
                rep = rep.flatten()

            representations[layer_name] = rep

            # Get metadata from first layer
            if metadata is None and self.return_metadata:
                metadata = {
                    "object": item["object"],
                    "style": item["style"],
                    "prompt_nr": item["prompt_nr"],
                    "prompt_text": item["prompt_text"],
                }

        if not self.return_metadata:
            return representations

        return representations, metadata
