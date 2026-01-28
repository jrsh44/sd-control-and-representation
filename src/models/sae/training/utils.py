"""
Utility functions for SAE training.
"""

from typing import Any, Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.models.sae.training.config import SchedulerConfig


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: SchedulerConfig,
    total_steps: int,
) -> Optional[SequentialLR]:
    """
    Create a learning rate scheduler with linear warmup and cosine annealing.

    The scheduler performs:
    1. Linear warmup: LR increases from (start_factor * base_lr) to base_lr
       over warmup_steps batches
    2. Cosine annealing: LR decreases from base_lr to (min_lr_ratio * base_lr)
       following a cosine curve over the remaining steps

    Args:
        optimizer: The optimizer to schedule
        scheduler_config: Configuration for the scheduler
        total_steps: Total number of training steps (batches)

    Returns:
        SequentialLR scheduler combining warmup and cosine phases,
        or None if scheduler is disabled
    """
    if not scheduler_config.enabled:
        return None

    warmup_steps = scheduler_config.warmup_steps
    if warmup_steps <= 0:
        return None

    original_warmup_steps = warmup_steps
    warmup_steps = min(warmup_steps, total_steps - 1)
    if warmup_steps < original_warmup_steps:
        print(f"  ⚠️  WARNING: warmup_steps ({original_warmup_steps}) > total_steps ({total_steps})")
        print(f"     Clamping warmup_steps to {warmup_steps}")
    cosine_steps = total_steps - warmup_steps

    base_lr = optimizer.param_groups[0]["lr"]
    eta_min = base_lr * scheduler_config.min_lr_ratio

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=scheduler_config.warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=eta_min,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    return scheduler


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
