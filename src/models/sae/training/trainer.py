"""
SAE Trainer class for structured training with callbacks and logging.
"""

import time
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import torch
from overcomplete.metrics import l2
from overcomplete.sae.trackers import DeadCodeTracker
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from .config import EpochMetrics, TrainingConfig, TrainingState
from .losses import criterion_laux
from .metrics import (
    MetricsAggregator,
    compute_dictionary_metrics,
    compute_encoder_decoder_similarity,
    compute_gradient_metrics,
    compute_reconstruction_error,
    compute_sparsity_metrics,
)
from .utils import extract_input, get_dictionary

warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
    category=UserWarning,
)


class SAETrainer:
    """
    Trainer class for Sparse Autoencoders.

    Handles training loop, validation, metrics collection, and callbacks.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        base_lr: Optional[float] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: SAE model to train
            optimizer: Optimizer for updating weights
            config: Training configuration
            scheduler: Optional learning rate scheduler
            base_lr: Base learning rate (before scheduler modifications).
                     If None, uses current optimizer LR.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scheduler = scheduler

        # Store base learning rate (before scheduler modifies it)
        # If not provided, use current optimizer LR
        self.base_lr = base_lr if base_lr is not None else optimizer.param_groups[0]["lr"]

        self.device = (
            torch.device(config.device) if isinstance(config.device, str) else config.device
        )

        self.scaler: Optional[GradScaler] = (
            GradScaler("cuda") if config.use_amp and self.device.type == "cuda" else None
        )

        self.dead_tracker: Optional[DeadCodeTracker] = None
        self.state = TrainingState()

        self._epoch_callbacks: List[Callable] = []
        self._batch_callbacks: List[Callable] = []

    def add_epoch_callback(self, callback: Callable):
        """
        Add callback called after each epoch.

        Args:
            callback: Function with signature
                (model, optimizer, epoch, train_metrics, val_metrics,
                 train_logs, val_logs) -> None
        """
        self._epoch_callbacks.append(callback)

    def add_batch_callback(self, callback: Callable):
        """
        Add callback called after each batch.

        Args:
            callback: Function with signature
                (step, batch_idx, epoch, metrics) -> None
        """
        self._batch_callbacks.append(callback)

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        existing_logs: Optional[Dict] = None,
    ) -> Dict:
        """
        Run the full training loop.

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation
            existing_logs: Pre-existing logs for resuming training

        Returns:
            Dictionary with 'train' and optionally 'val' logs
        """
        if existing_logs:
            self.state.train_logs = defaultdict(list, existing_logs.get("train", {}))
            if val_dataloader is not None:
                self.state.val_logs = defaultdict(list, existing_logs.get("val", {}))
            print(f"   Resuming from epoch {self.config.start_epoch + 1}")

        self._print_config(train_dataloader, val_dataloader)

        for epoch in range(self.config.start_epoch, self.config.nb_epochs):
            self.state.current_epoch = epoch

            # Set epoch on sampler if it has set_epoch method (for reproducible shuffling)
            if hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)

            train_metrics = self._train_epoch(train_dataloader, epoch)
            self._log_epoch_metrics("train", train_metrics)

            val_metrics = None
            if val_dataloader is not None:
                val_metrics = self._validate_epoch(val_dataloader)
                self._log_epoch_metrics("val", val_metrics)

                if val_metrics.loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_metrics.loss

            for callback in self._epoch_callbacks:
                callback(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    train_logs=dict(self.state.train_logs),
                    val_logs=dict(self.state.val_logs) if val_dataloader else None,
                )

            print()

        self._print_summary(val_dataloader is not None)

        result = {"train": dict(self.state.train_logs)}
        if val_dataloader is not None:
            result["val"] = dict(self.state.val_logs)
        return result

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> EpochMetrics:
        """
        Run one training epoch.

        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            EpochMetrics dataclass with training metrics
        """
        print(f"Epoch [{epoch + 1}/{self.config.nb_epochs}]")
        print("-" * 80)
        print("🏋️  Training phase...")

        self.model.train()
        metrics_agg = MetricsAggregator()
        start_time = time.time()
        last_logged_progress = 0

        # Timing accumulators
        total_data_loading_time = 0.0
        total_batch_compute_time = 0.0
        data_load_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Record data loading time
            data_load_end = time.time()
            total_data_loading_time += data_load_end - data_load_start

            # Time the batch computation
            compute_start = time.time()
            batch_metrics = self._train_step(batch, batch_idx)
            compute_end = time.time()
            total_batch_compute_time += compute_end - compute_start

            # Add timing to batch metrics
            batch_metrics["data_loading_time"] = data_load_end - data_load_start
            batch_metrics["batch_compute_time"] = compute_end - compute_start

            metrics_agg.update(batch_metrics)
            self.state.global_step += 1

            for callback in self._batch_callbacks:
                callback(
                    step=self.state.global_step,
                    batch_idx=batch_idx,
                    epoch=epoch,
                    metrics=batch_metrics,
                )

            progress = (batch_idx + 1) / len(dataloader)
            progress_percent = int(progress * 100)

            if progress_percent >= last_logged_progress + 10 or batch_idx == len(dataloader) - 1:
                self._log_progress(batch_idx, len(dataloader), start_time, batch_metrics)
                last_logged_progress = progress_percent

            # Start timing next data load
            data_load_start = time.time()

        elapsed = time.time() - start_time
        avg_metrics = metrics_agg.get_averages()
        num_batches = len(dataloader)

        # Collect parameter/gradient norms at end of epoch
        params_norms = {}
        grad_norms = {}
        for name, param in self.model.named_parameters():
            params_norms[name] = l2(param).item()
            if param.grad is not None:
                grad_norms[name] = l2(param.grad).item()

        epoch_metrics = EpochMetrics(
            loss=avg_metrics.get("loss", 0),
            recon_loss=avg_metrics.get("recon_loss", 0),
            aux_loss=avg_metrics.get("aux_loss", 0),
            r2=avg_metrics.get("r2", 0),
            l0_sparsity=avg_metrics.get("l0_sparsity", 0),
            z_l2=avg_metrics.get("z_l2", 0),
            dead_features_ratio=(self.dead_tracker.get_dead_ratio() if self.dead_tracker else 0),
            time_seconds=elapsed,
            num_batches=num_batches,
            learning_rate=self.optimizer.param_groups[0]["lr"],
            mean_activation=avg_metrics.get("mean_activation", 0),
            max_activation=avg_metrics.get("max_activation", 0),
            dict_sparsity=avg_metrics.get("dict_sparsity", 0),
            dict_norms_mean=avg_metrics.get("dict_norms_mean", 0),
            data_loading_time=total_data_loading_time,
            batch_compute_time=total_batch_compute_time,
            avg_data_loading_time=total_data_loading_time / num_batches if num_batches > 0 else 0,
            avg_batch_compute_time=total_batch_compute_time / num_batches if num_batches > 0 else 0,
            params_norms=params_norms,
            grad_norms=grad_norms,
        )

        print(f"   ✅ Training complete in {elapsed:.2f}s")
        print(
            f"   📊 Loss: {epoch_metrics.loss:.4f} "
            f"(recon: {epoch_metrics.recon_loss:.4f}, aux: {epoch_metrics.aux_loss:.4f}) | "
            f"R²: {epoch_metrics.r2:.4f} | "
            f"L0: {epoch_metrics.l0_sparsity:.2f} | "
            f"Dead: {epoch_metrics.dead_features_ratio * 100:.1f}%"
        )
        print(
            f"   ⏱️  Timing: data={total_data_loading_time:.2f}s "
            f"({epoch_metrics.avg_data_loading_time * 1000:.1f}ms/batch), "
            f"compute={total_batch_compute_time:.2f}s "
            f"({epoch_metrics.avg_batch_compute_time * 1000:.1f}ms/batch)"
        )

        return epoch_metrics

    def _train_step(self, batch: Any, batch_idx: int) -> Dict:
        """
        Run one training step.

        Args:
            batch: Input batch from DataLoader
            batch_idx: Index of the current batch

        Returns:
            Dictionary of metrics for the batch
        """
        x = extract_input(batch)
        if x is None:
            raise ValueError("Batch input cannot be None")
        x = x.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        dictionary = get_dictionary(self.model)

        if self.scaler is not None:
            with autocast("cuda"):
                pre_codes, codes, x_hat = self.model(x)
                recon_loss, aux_loss = criterion_laux(x, x_hat, pre_codes, codes, dictionary)
                loss = recon_loss + self.config.aux_loss_alpha * aux_loss

            self.scaler.scale(loss).backward()
            if self.config.clip_grad:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            pre_codes, codes, x_hat = self.model(x)
            recon_loss, aux_loss = criterion_laux(x, x_hat, pre_codes, codes, dictionary)
            loss = recon_loss + self.config.aux_loss_alpha * aux_loss
            loss.backward()
            if self.config.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        if self.dead_tracker is None:
            self.dead_tracker = DeadCodeTracker(codes.shape[1], self.device)
            print(f"   Initialized dead code tracker ({codes.shape[1]} features)")

        metrics = {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "aux_loss": aux_loss.item(),
        }

        should_compute_expensive = batch_idx % self.config.log_interval == 0

        if should_compute_expensive:
            self.dead_tracker.update(codes.detach())

            if self.config.compute_expensive_metrics:
                with torch.no_grad():
                    metrics["r2"] = compute_reconstruction_error(x, x_hat)

                    sparsity_metrics = compute_sparsity_metrics(codes.detach())
                    metrics["l0_sparsity"] = sparsity_metrics["l0_sparsity"]
                    metrics["z_l2"] = sparsity_metrics["z_l2"]
                    metrics["mean_activation"] = sparsity_metrics["mean_activation"]
                    metrics["max_activation"] = sparsity_metrics["max_activation"]

                    dict_metrics = compute_dictionary_metrics(dictionary)
                    metrics["dict_sparsity"] = dict_metrics["sparsity"]
                    metrics["dict_norms_mean"] = dict_metrics["norms_mean"]

            if self.config.compute_gradient_metrics:
                grad_metrics = compute_gradient_metrics(self.model)
                metrics.update(grad_metrics)

        return metrics

    def _validate_epoch(self, dataloader: DataLoader) -> EpochMetrics:
        """
        Run one validation epoch with comprehensive metrics.

        Args:
            dataloader: DataLoader for validation data

        Returns:
            EpochMetrics dataclass with validation metrics
        """
        print("   🔍 Validation phase...")

        self.model.eval()
        metrics_agg = MetricsAggregator()
        start_time = time.time()
        last_logged_progress = 0
        val_dead_tracker = None

        # Timing accumulators
        total_data_loading_time = 0.0
        total_batch_compute_time = 0.0
        data_load_start = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Record data loading time
                data_load_end = time.time()
                total_data_loading_time += data_load_end - data_load_start

                compute_start = time.time()

                x = extract_input(batch)
                if x is None:
                    raise ValueError("Batch input cannot be None")
                x = x.to(self.device, non_blocking=True)

                dictionary = get_dictionary(self.model)

                if self.scaler is not None:
                    with autocast("cuda"):
                        pre_codes, codes, x_hat = self.model(x)
                        recon_loss, aux_loss = criterion_laux(
                            x, x_hat, pre_codes, codes, dictionary
                        )
                        loss = recon_loss + self.config.aux_loss_alpha * aux_loss
                else:
                    pre_codes, codes, x_hat = self.model(x)
                    recon_loss, aux_loss = criterion_laux(x, x_hat, pre_codes, codes, dictionary)
                    loss = recon_loss + self.config.aux_loss_alpha * aux_loss

                # Initialize trackers
                if val_dead_tracker is None:
                    val_dead_tracker = DeadCodeTracker(codes.shape[1], self.device)

                val_dead_tracker.update(codes)

                metrics = {
                    "loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "aux_loss": aux_loss.item(),
                    "r2": compute_reconstruction_error(x, x_hat),
                }

                sparsity_metrics = compute_sparsity_metrics(codes)
                metrics["l0_sparsity"] = sparsity_metrics["l0_sparsity"]
                metrics["z_l2"] = sparsity_metrics["z_l2"]
                metrics["mean_activation"] = sparsity_metrics["mean_activation"]

                compute_end = time.time()
                total_batch_compute_time += compute_end - compute_start

                metrics_agg.update(metrics)

                progress = (batch_idx + 1) / len(dataloader)
                progress_percent = int(progress * 100)

                if (
                    progress_percent >= last_logged_progress + 10
                    or batch_idx == len(dataloader) - 1
                ):
                    self._log_progress(batch_idx, len(dataloader), start_time, metrics)
                    last_logged_progress = progress_percent

                # Start timing next data load
                data_load_start = time.time()

        elapsed = time.time() - start_time
        avg_metrics = metrics_agg.get_averages()
        num_batches = len(dataloader)

        # Compute similarity metrics at end of validation
        similarity_metrics = compute_encoder_decoder_similarity(self.model)

        epoch_metrics = EpochMetrics(
            loss=avg_metrics.get("loss", 0),
            recon_loss=avg_metrics.get("recon_loss", 0),
            aux_loss=avg_metrics.get("aux_loss", 0),
            r2=avg_metrics.get("r2", 0),
            l0_sparsity=avg_metrics.get("l0_sparsity", 0),
            z_l2=avg_metrics.get("z_l2", 0),
            dead_features_ratio=(val_dead_tracker.get_dead_ratio() if val_dead_tracker else 0),
            time_seconds=elapsed,
            num_batches=num_batches,
            mean_activation=avg_metrics.get("mean_activation", 0),
            encoder_avg_max_cos=similarity_metrics.get("encoder_avg_max_cos", 0),
            decoder_avg_max_cos=similarity_metrics.get("decoder_avg_max_cos", 0),
            decoder_mean_norm=similarity_metrics.get("decoder_mean_norm", 0),
            data_loading_time=total_data_loading_time,
            batch_compute_time=total_batch_compute_time,
            avg_data_loading_time=total_data_loading_time / num_batches if num_batches > 0 else 0,
            avg_batch_compute_time=total_batch_compute_time / num_batches if num_batches > 0 else 0,
        )

        print(
            f"   📊 Val Loss: {epoch_metrics.loss:.4f} "
            f"(recon: {epoch_metrics.recon_loss:.4f}, aux: {epoch_metrics.aux_loss:.4f}) | "
            f"R²: {epoch_metrics.r2:.4f} | "
            f"L0: {epoch_metrics.l0_sparsity:.2f} | "
            f"Dead: {epoch_metrics.dead_features_ratio * 100:.1f}%"
        )
        print(
            f"   📊 Similarity - Encoder: {epoch_metrics.encoder_avg_max_cos:.4f}, "
            f"Decoder: {epoch_metrics.decoder_avg_max_cos:.4f}, "
            f"Decoder Norm: {epoch_metrics.decoder_mean_norm:.4f}"
        )
        print(
            f"   ⏱️  Timing: data={total_data_loading_time:.2f}s "
            f"({epoch_metrics.avg_data_loading_time * 1000:.1f}ms/batch), "
            f"compute={total_batch_compute_time:.2f}s "
            f"({epoch_metrics.avg_batch_compute_time * 1000:.1f}ms/batch)"
        )

        return epoch_metrics

    def _log_epoch_metrics(self, phase: str, metrics: EpochMetrics):
        """Log epoch metrics to internal state."""
        logs = self.state.train_logs if phase == "train" else self.state.val_logs

        # Core metrics
        logs["avg_loss"].append(metrics.loss)
        logs["recon_loss"].append(metrics.recon_loss)
        logs["aux_loss"].append(metrics.aux_loss)
        logs["r2"].append(metrics.r2)
        logs["z_sparsity"].append(metrics.l0_sparsity)
        logs["z_l2"].append(metrics.z_l2)
        logs["dead_features"].append(metrics.dead_features_ratio)
        logs["time_epoch"].append(metrics.time_seconds)
        logs["learning_rate"].append(metrics.learning_rate)
        logs["mean_activation"].append(metrics.mean_activation)

        # Timing metrics
        logs["data_loading_time"].append(metrics.data_loading_time)
        logs["batch_compute_time"].append(metrics.batch_compute_time)
        logs["avg_data_loading_time_ms"].append(metrics.avg_data_loading_time * 1000)
        logs["avg_batch_compute_time_ms"].append(metrics.avg_batch_compute_time * 1000)

        if phase == "train":
            logs["dict_sparsity"].append(metrics.dict_sparsity)
            logs["dict_norms_mean"].append(metrics.dict_norms_mean)

            # Log parameter/gradient norms
            if metrics.params_norms:
                for name, value in metrics.params_norms.items():
                    logs[f"params_norm/{name}"].append(value)
            if metrics.grad_norms:
                for name, value in metrics.grad_norms.items():
                    logs[f"grad_norm/{name}"].append(value)
        else:
            # Validation-specific metrics
            logs["encoder_avg_max_cos"].append(metrics.encoder_avg_max_cos)
            logs["decoder_avg_max_cos"].append(metrics.decoder_avg_max_cos)
            logs["decoder_mean_norm"].append(metrics.decoder_mean_norm)

    def _log_progress(
        self,
        batch_idx: int,
        total_batches: int,
        start_time: float,
        metrics: Dict,
    ):
        """Log progress during training/validation."""
        elapsed = time.time() - start_time
        avg_time = elapsed / (batch_idx + 1)
        remaining = avg_time * (total_batches - batch_idx - 1)
        eta_min = int(remaining // 60)
        eta_sec = int(remaining % 60)

        progress = int((batch_idx + 1) / total_batches * 100)
        loss = metrics.get("loss", 0)

        print(
            f"   [{progress:3d}%] Batch {batch_idx + 1}/{total_batches} | "
            f"Loss: {loss:.4f} | "
            f"ETA: {eta_min}m {eta_sec:02d}s ({avg_time:.2f}s/batch)"
        )

    def _print_config(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader]):
        """Print training configuration."""
        print("\n" + "=" * 80)
        print("SAE TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"Epochs: {self.config.nb_epochs}")
        print(f"Training batches: {len(train_dataloader)}")
        if val_dataloader:
            print(f"Validation batches: {len(val_dataloader)}")
        print(f"Gradient clipping: {self.config.clip_grad}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {'Enabled' if self.scaler else 'Disabled'}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        current_lr = self.optimizer.param_groups[0]["lr"]
        if self.scheduler is not None and current_lr != self.base_lr:
            print(
                f"Learning rate: {self.base_lr} (base), {current_lr:.2e} (initial after warmup start)"
            )
        else:
            print(f"Learning rate: {current_lr}")
        print(f"Aux loss alpha: {self.config.aux_loss_alpha}")
        print(f"Log interval: {self.config.log_interval} batches")
        if self.config.start_epoch > 0:
            print(f"Resuming from: Epoch {self.config.start_epoch + 1}")
        print("=" * 80 + "\n")

    def _print_summary(self, has_validation: bool):
        """Print training summary."""
        train_logs = self.state.train_logs

        print("\n" + "=" * 80)
        print("SAE TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total epochs: {self.config.nb_epochs}")
        total_time = sum(train_logs["time_epoch"])
        print(f"Total time: {total_time:.2f}s")
        avg_time = total_time / len(train_logs["time_epoch"])
        print(f"Avg time/epoch: {avg_time:.2f}s")
        print()
        print("Final Training Metrics:")
        print(f"  Loss: {train_logs['avg_loss'][-1]:.4f}")
        print(f"    Recon: {train_logs['recon_loss'][-1]:.4f}")
        print(f"    Aux: {train_logs['aux_loss'][-1]:.4f}")
        print(f"  R²: {train_logs['r2'][-1]:.4f}")
        print(f"  L0 Sparsity: {train_logs['z_sparsity'][-1]:.2f}")
        print(f"  Z L2: {train_logs['z_l2'][-1]:.4f}")
        print(f"  Dead Features: {train_logs['dead_features'][-1] * 100:.1f}%")

        if has_validation:
            val_logs = self.state.val_logs
            print()
            print("Final Validation Metrics:")
            print(f"  Loss: {val_logs['avg_loss'][-1]:.4f}")
            print(f"    Recon: {val_logs['recon_loss'][-1]:.4f}")
            print(f"    Aux: {val_logs['aux_loss'][-1]:.4f}")
            print(f"  R²: {val_logs['r2'][-1]:.4f}")
            print(f"  L0 Sparsity: {val_logs['z_sparsity'][-1]:.2f}")
            print(f"  Dead Features: {val_logs['dead_features'][-1] * 100:.1f}%")
            print(f"  Best Val Loss: {self.state.best_val_loss:.4f}")
            print()
            print("  Similarity Metrics:")
            print(f"    Encoder Avg Max Cos: {val_logs['encoder_avg_max_cos'][-1]:.4f}")
            print(f"    Decoder Avg Max Cos: {val_logs['decoder_avg_max_cos'][-1]:.4f}")
            print(f"    Decoder Mean Norm: {val_logs['decoder_mean_norm'][-1]:.4f}")
        print("=" * 80 + "\n")
