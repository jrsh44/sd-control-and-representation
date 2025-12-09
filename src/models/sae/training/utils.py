"""
Utility functions for SAE training.
"""

from typing import Any, Optional

import torch


def extract_input(batch: Any) -> Optional[torch.Tensor]:
    """
    Extract input tensor from various batch formats.

    Supports:
    - Tuple/list: Returns first element
    - Dict: Returns value for 'data' key
    - Tensor: Returns as-is

    Args:
        batch: Batch data in various formats

    Returns:
        Input tensor or None if extraction fails
    """
    if isinstance(batch, (tuple, list)):
        return batch[0]
    elif isinstance(batch, dict):
        return batch.get("data")
    return batch


def get_dictionary(model: torch.nn.Module) -> torch.Tensor:
    """
    Get dictionary from model, handling both property and method access.

    Args:
        model: SAE model with get_dictionary attribute

    Returns:
        Dictionary tensor
    """
    dictionary = model.get_dictionary
    if callable(dictionary):
        return dictionary()
    return dictionary
