"""
Configuration dataclasses for SAE training.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class SchedulerConfig:
    """
    Configuration for learning rate scheduler with warmup and cosine annealing.
    """

    enabled: bool = False
    warmup_steps: int = 0  # Number of batches for warmup phase
    warmup_start_factor: float = 0.01  # Start at 1% of base LR
    min_lr_ratio: float = 0.0  # Minimum LR as ratio of base LR (eta_min)

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the config.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SchedulerConfig":
        """
        Create from dictionary.

        Args:
            d: Dictionary with config values.

        Returns:
            SchedulerConfig instance.
        """
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Configuration for SAE training."""

    nb_epochs: int = 20
    clip_grad: float = 1.0
    use_amp: bool = True
    log_interval: int = 10
    device: str = "cuda"
    start_epoch: int = 0
    compute_expensive_metrics: bool = True
    compute_gradient_metrics: bool = False
    aux_loss_alpha: float = 1 / 32  # Scaling factor for auxiliary loss


@dataclass
class EpochMetrics:
    """Metrics collected during one epoch."""

    # Loss components
    loss: float = 0.0
    recon_loss: float = 0.0
    aux_loss: float = 0.0

    # Reconstruction quality
    r2: float = 0.0

    # Sparsity metrics
    l0_sparsity: float = 0.0
    z_l2: float = 0.0
    mean_activation: float = 0.0
    max_activation: float = 0.0
    active_ratio: float = 0.0

    # Dead features
    dead_features_ratio: float = 0.0

    # Dictionary metrics
    dict_sparsity: float = 0.0
    dict_norms_mean: float = 0.0

    # Similarity metrics (validation only)
    encoder_avg_max_cos: float = 0.0
    decoder_avg_max_cos: float = 0.0
    decoder_mean_norm: float = 0.0

    # Active features at thresholds (validation only)
    active_features_0_5: int = 0
    active_features_0_4: int = 0
    active_features_0_3: int = 0
    active_features_0_2: int = 0
    active_features_0_1: int = 0

    # Timing and metadata
    time_seconds: float = 0.0
    num_batches: int = 0
    learning_rate: float = 0.0

    # Timing breakdown
    data_loading_time: float = 0.0  # Total time waiting for data
    batch_compute_time: float = 0.0  # Total time for forward/backward/optim
    avg_data_loading_time: float = 0.0  # Average per batch
    avg_batch_compute_time: float = 0.0  # Average per batch

    # Parameter/gradient norms (optional)
    params_norms: Optional[Dict[str, float]] = None
    grad_norms: Optional[Dict[str, float]] = None


@dataclass
class TrainingState:
    """Mutable state during training."""

    current_epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    train_logs: Dict[str, List] = field(default_factory=lambda: defaultdict(list))
    val_logs: Dict[str, List] = field(default_factory=lambda: defaultdict(list))
