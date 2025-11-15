import time
from collections import defaultdict

import torch
from einops import rearrange
from overcomplete.metrics import l0_eps, l2, r2_score
from overcomplete.sae.trackers import DeadCodeTracker
from torch.amp.grad_scaler import GradScaler

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def criterion_laux(x, x_hat, pre_codes, codes, dictionary):
    """
    Custom criterion function for Sparse Autoencoder (SAE) training.
    Args:
        x (torch.Tensor): Original input tensor.
        x_hat (torch.Tensor): Reconstructed output tensor from the SAE.
        pre_codes (torch.Tensor): Pre-activation codes from the encoder.
        codes (torch.Tensor): Final sparse codes after applying top-k sparsity.
        dictionary (torch.Tensor): The learned dictionary (decoder weights).
    n (int): Dimensionality of the input features.
    Returns:
        torch.Tensor: The computed loss value.
    """
    n = x.shape[1]

    # Compute reconstruction MSE
    mse = torch.mean((x - x_hat) ** 2)

    # Hyperparameters (these could be passed as arguments or globals)
    alpha = 1 / 32  # Scaling factor for auxiliary loss
    k_aux = n // 2  # Number of least active features for auxiliary loss

    # Compute mean activation per feature across the batch
    feature_acts = codes.mean(dim=0)

    # Select indices of k_aux features with the lowest mean activations (potential dead latents)
    low_act_indices = torch.topk(feature_acts, k_aux, largest=False)[1]

    # Create a masked codes tensor using only the low-activation features
    codes_aux = torch.zeros_like(codes)
    codes_aux[:, low_act_indices] = codes[:, low_act_indices]

    # Reconstruct using only these features
    x_aux = torch.matmul(codes_aux, dictionary)

    # Compute auxiliary MSE on this partial reconstruction
    l_aux = torch.mean((x - x_aux) ** 2)

    # Total metric: MSE + alpha * L_aux
    metric = mse + alpha * l_aux

    return metric


def extract_input(batch) -> torch.Tensor:
    """
    Extracts the input data from a batch. Supports different batch types:
    - If the batch is a tuple or list, returns the first element.
    - If the batch is a dict, tries to return the value with key 'data' (or 'input').
    - Otherwise, assumes the batch is the input data.
    """
    if isinstance(batch, (tuple, list)):
        return batch[0]
    elif isinstance(batch, dict):
        data = batch.get("data")
        if data is None:
            raise ValueError("Batch dict does not contain 'data' key")
        return data
    else:
        return batch


def _compute_reconstruction_error(x, x_hat):
    """
    Try to match the shapes of x and x_hat to compute the reconstruction error.

    If the input (x) shape is 4D assume it is (n, c, w, h), if it is 3D assume
    it is (n, t, c). Else, assume it is already flattened.

    Concerning the reconstruction error, we want a measure that could be compared
    across different input shapes, so we use the R2 score: 1 means perfect
    reconstruction, 0 means no reconstruction at all.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.

    Returns
    -------
    torch.Tensor
        Reconstruction error (R2 score) as a tensor on the same device.
    """
    if len(x.shape) == 4 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n c w h -> (n w h) c")
    elif len(x.shape) == 3 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n t c -> (n t) c")
    else:
        if x.shape != x_hat.shape:
            raise ValueError("Input and output shapes must match.")
        x_flatten = x

    r2 = r2_score(x_flatten, x_hat)

    return r2


def _log_metrics(monitoring, logs, model, z, loss, optimizer):
    """
    Log training metrics for the current training step.

    Parameters
    ----------
    monitoring : int
        Monitoring level, for 1 store only basic statistics, for 2 store more detailed statistics.
    logs : defaultdict
        Logs of training statistics.
    model : nn.Module
        The SAE model.
    z : torch.Tensor
        Encoded tensor.
    loss : torch.Tensor
        Loss value.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    """
    if monitoring == 0:
        return

    if monitoring > 0:
        logs["lr"].append(optimizer.param_groups[0]["lr"])
        logs["step_loss"].append(loss.item())

    if monitoring > 1:
        # store directly some z values
        # and the params / gradients norms
        logs["z"].append(z.detach()[::10])
        logs["z_l2"].append(l2(z).item())

        logs["dictionary_sparsity"].append(l0_eps(model.get_dictionary()).mean().item())
        logs["dictionary_norms"].append(l2(model.get_dictionary(), -1).mean().item())

        for name, param in model.named_parameters():
            if param.grad is not None:
                logs[f"params_norm_{name}"].append(l2(param).item())
                logs[f"params_grad_norm_{name}"].append(l2(param.grad).item())


def train_sae_val(
    model,
    train_dataloader,
    criterion,
    optimizer,
    val_dataloader=None,
    scheduler=None,
    nb_epochs=20,
    clip_grad=1.0,
    monitoring=1,
    device="cpu",
    use_amp=True,
    log_interval=10,
    wandb_enabled=False,
):
    """
    Train a Sparse Autoencoder (SAE) model with optional validation set.

    Parameters
    ----------
    model : nn.Module
        The SAE model to train.
    train_dataloader : DataLoader
        DataLoader for training data.
    criterion : callable
        Loss function.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    val_dataloader : DataLoader, optional
        DataLoader for validation data. If None, no validation is performed.
    scheduler : callable, optional
        Learning rate scheduler.
    nb_epochs : int, optional
        Number of training epochs.
    clip_grad : float, optional
        Gradient clipping value.
    monitoring : int, optional
        Monitoring level: 0=silent, 1=loss+R2+L0+dead, 2=gradients too.
    device : str or torch.device, optional
        Device to run on.
    use_amp : bool, optional
        Use automatic mixed precision training (default: True).
    log_interval : int, optional
        Update progress every N batches (default: 10).
    wandb_enabled : bool, optional
        Enable wandb logging (default: False).

    Returns
    -------
    dict
        Dictionary with keys:
            - 'train': logs for training
            - 'val': logs for validation (empty if val_dataloader is None)
    """
    # Convert device to torch.device if it's a string
    if isinstance(device, str):
        device = torch.device(device)

    train_logs = defaultdict(list)
    val_logs = defaultdict(list) if val_dataloader is not None else None

    # Initialize AMP scaler for mixed precision (only on CUDA)
    scaler = GradScaler("cuda") if use_amp and device.type == "cuda" else None

    # Initialize dead code tracker on first batch
    dead_tracker = None

    # Print training configuration
    print("\n" + "=" * 80)
    print("SAE TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Epochs: {nb_epochs}")
    print(f"Training batches: {len(train_dataloader)}")
    if val_dataloader is not None:
        print(f"Validation batches: {len(val_dataloader)}")
    print(f"Gradient clipping: {clip_grad}")
    print(f"Device: {device}")
    print(f"Mixed Precision: {'Enabled' if scaler is not None else 'Disabled'}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Progress updates: Every {log_interval} batches")
    print("=" * 80 + "\n")

    for epoch in range(nb_epochs):
        # ==================== TRAINING PHASE ====================
        print(f"Epoch [{epoch + 1}/{nb_epochs}]")
        print("-" * 80)
        print("🏋️  Training phase...")

        model.train()
        start_time = time.time()

        epoch_loss_tensor = torch.tensor(0.0, device=device)
        epoch_error_tensor = torch.tensor(0.0, device=device)
        epoch_sparsity_tensor = torch.tensor(0.0, device=device)
        batch_count = 0
        metric_samples = 0

        for batch_idx, batch in enumerate(train_dataloader):
            batch_start = time.time()
            batch_count += 1

            # Measure data loading time (DataLoader gives us the batch)
            # This measures how long it took DataLoader workers to fetch all items
            data_load_start = time.time()
            x = extract_input(batch).to(device, non_blocking=True)
            data_load_time = time.time() - data_load_start

            # Note: The actual time to load batch items happens in DataLoader workers
            # before this point. This only measures the transfer to GPU.

            optimizer.zero_grad(set_to_none=True)

            # Use automatic mixed precision
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    z_pre, z, x_hat = model(x)
                    loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())

                scaler.scale(loss).backward()
                if clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                z_pre, z, x_hat = model(x)
                loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Initialize dead tracker
            if dead_tracker is None:
                dead_tracker = DeadCodeTracker(z.shape[1], device)
                print(f"   Initialized dead code tracker (tracking {z.shape[1]} features)")

            # Update dead tracker periodically
            if batch_idx % log_interval == 0 and dead_tracker is not None:
                dead_tracker.update(z.detach())

            if monitoring:
                epoch_loss_tensor += loss.detach()
                # Compute expensive metrics periodically
                if batch_idx % log_interval == 0:
                    metric_samples += 1
                    with torch.no_grad():
                        epoch_error_tensor += _compute_reconstruction_error(x, x_hat)
                        epoch_sparsity_tensor += l0_eps(z.detach(), 0).sum()

                # Log detailed metrics periodically
                if monitoring > 1 and batch_idx % log_interval == 0:
                    _log_metrics(monitoring, train_logs, model, z.detach(), loss, optimizer)

            # Synchronize GPU before timing (get real execution time, not just CPU scheduling time)
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Track batch time and show immediate feedback
            batch_time = time.time() - batch_start
            elapsed = time.time() - start_time
            avg_time_per_batch = elapsed / (batch_idx + 1)
            remaining_batches = len(train_dataloader) - (batch_idx + 1)
            eta_seconds = avg_time_per_batch * remaining_batches
            eta_minutes = int(eta_seconds // 60)
            eta_secs = int(eta_seconds % 60)

            progress_percent = int(((batch_idx + 1) / len(train_dataloader)) * 100)

            # Only show progress every log_interval batches to avoid overhead
            if batch_idx % log_interval == 0 or batch_idx == len(train_dataloader) - 1:
                compute_time = batch_time - data_load_time
                print(
                    f"   [{progress_percent:3d}%] "
                    f"Batch {batch_idx + 1}/{len(train_dataloader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Total: {batch_time:.2f}s "
                    f"(Load: {data_load_time:.3f}s, Compute: {compute_time:.2f}s) | "
                    f"ETA: {eta_minutes}m {eta_secs:02d}s"
                )

        train_time = time.time() - start_time

        # ==================== TRAINING METRICS ====================
        if monitoring and batch_count > 0:
            train_logs["avg_loss"].append((epoch_loss_tensor / batch_count).item())
            # Adjust for periodic sampling
            if metric_samples > 0:
                train_logs["r2"].append((epoch_error_tensor / metric_samples).item())
                train_logs["z_sparsity"].append((epoch_sparsity_tensor / metric_samples).item())
            if dead_tracker is not None:
                train_logs["dead_features"].append(dead_tracker.get_dead_ratio())
            train_logs["time_epoch"].append(train_time)

            print(f"   ✅ Training complete in {train_time:.2f}s")
            print(
                f"   📊 Train Loss: {train_logs['avg_loss'][-1]:.4f} | "
                f"R²: {train_logs['r2'][-1]:.4f} | "
                f"L0: {train_logs['z_sparsity'][-1]:.2f} | "
                f"Dead: {train_logs['dead_features'][-1] * 100:.1f}%"
            )

        # ==================== VALIDATION PHASE ====================
        if val_dataloader is not None:
            print("   🔍 Validation phase...")
            model.eval()

            val_loss_tensor = torch.tensor(0.0, device=device)
            val_error_tensor = torch.tensor(0.0, device=device)
            val_sparsity_tensor = torch.tensor(0.0, device=device)
            val_batch_count = 0

            if dead_tracker is not None:
                val_dead_tracker = DeadCodeTracker(z.shape[1], device)
            else:
                val_dead_tracker = None

            val_start_time = time.time()

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    val_batch_start = time.time()
                    val_batch_count += 1

                    # Measure data loading time
                    val_data_load_start = time.time()
                    x = extract_input(batch).to(device, non_blocking=True)
                    val_data_load_time = time.time() - val_data_load_start

                    # Use AMP for validation too
                    if scaler is not None:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            z_pre, z, x_hat = model(x)
                            loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())
                    else:
                        z_pre, z, x_hat = model(x)
                        loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())

                    val_loss_tensor += loss.detach()
                    val_error_tensor += _compute_reconstruction_error(x, x_hat)
                    val_sparsity_tensor += l0_eps(z, 0).sum()

                    if val_dead_tracker is not None:
                        val_dead_tracker.update(z)

                    # Synchronize GPU before timing to get real execution time
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    val_batch_time = time.time() - val_batch_start
                    val_elapsed = time.time() - val_start_time
                    avg_time_per_batch = val_elapsed / (batch_idx + 1)
                    remaining_batches = len(val_dataloader) - (batch_idx + 1)
                    eta_seconds = avg_time_per_batch * remaining_batches
                    eta_minutes = int(eta_seconds // 60)
                    eta_secs = int(eta_seconds % 60)

                    progress_percent = int(((batch_idx + 1) / len(val_dataloader)) * 100)

                    val_compute_time = val_batch_time - val_data_load_time
                    print(
                        f"   [{progress_percent:3d}%] "
                        f"Batch {batch_idx + 1}/{len(val_dataloader)} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Total: {val_batch_time:.2f}s "
                        f"(Load: {val_data_load_time:.3f}s, "
                        f"Compute: {val_compute_time:.2f}s) | "
                        f"ETA: {eta_minutes}m {eta_secs:02d}s"
                    )

            if val_batch_count > 0 and val_logs is not None:
                val_logs["avg_loss"].append((val_loss_tensor / val_batch_count).item())
                val_logs["r2"].append((val_error_tensor / val_batch_count).item())
                val_logs["z_sparsity"].append((val_sparsity_tensor / val_batch_count).item())
                val_logs["dead_features"].append(
                    val_dead_tracker.get_dead_ratio() if val_dead_tracker else 0.0
                )

                print(
                    f"   📊 Val Loss: {val_logs['avg_loss'][-1]:.4f} | "
                    f"R²: {val_logs['r2'][-1]:.4f} | "
                    f"L0: {val_logs['z_sparsity'][-1]:.2f} | "
                    f"Dead: {val_logs['dead_features'][-1] * 100:.1f}%"
                )

        if wandb_enabled and WANDB_AVAILABLE:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": (train_logs["avg_loss"][-1] if train_logs["avg_loss"] else 0.0),
                "train/r2": train_logs["r2"][-1] if train_logs["r2"] else 0.0,
                "train/l0_sparsity": (
                    train_logs["z_sparsity"][-1] if train_logs["z_sparsity"] else 0.0
                ),
                "train/dead_features_ratio": (
                    train_logs["dead_features"][-1] if train_logs["dead_features"] else 0.0
                ),
                "train/time_seconds": (
                    train_logs["time_epoch"][-1] if train_logs["time_epoch"] else 0.0
                ),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }

            if val_dataloader is not None and val_logs is not None:
                if val_logs["avg_loss"]:
                    log_dict.update(
                        {
                            "val/loss": val_logs["avg_loss"][-1],
                            "val/r2": val_logs["r2"][-1],
                            "val/l0_sparsity": val_logs["z_sparsity"][-1],
                            "val/dead_features_ratio": val_logs["dead_features"][-1],
                        }
                    )

            if device.type == "cuda":
                log_dict["gpu/memory_allocated_mb"] = torch.cuda.memory_allocated(device) / 1024**2
                log_dict["gpu/memory_reserved_mb"] = torch.cuda.memory_reserved(device) / 1024**2
                log_dict["gpu/max_memory_allocated_mb"] = (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                )

            wandb.log(log_dict)

        print()  # Empty line between epochs

    # ==================== TRAINING SUMMARY ====================
    print("\n" + "=" * 80)
    print("SAE TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total epochs completed: {nb_epochs}")
    print(f"Total training time: {sum(train_logs['time_epoch']):.2f}s")
    avg_time = sum(train_logs["time_epoch"]) / len(train_logs["time_epoch"])
    print(f"Average time per epoch: {avg_time:.2f}s")
    print()
    print("Final Training Metrics:")
    if train_logs["avg_loss"]:
        print(f"  Loss: {train_logs['avg_loss'][-1]:.4f}")
    if train_logs["r2"]:
        print(f"  R² Score: {train_logs['r2'][-1]:.4f}")
    if train_logs["z_sparsity"]:
        print(f"  L0 Sparsity: {train_logs['z_sparsity'][-1]:.2f}")
    if train_logs["dead_features"]:
        print(f"  Dead Features: {train_logs['dead_features'][-1] * 100:.1f}%")

    if val_dataloader is not None and val_logs is not None:
        print()
        print("Final Validation Metrics:")
        if val_logs["avg_loss"]:
            print(f"  Loss: {val_logs['avg_loss'][-1]:.4f}")
        if val_logs["r2"]:
            print(f"  R² Score: {val_logs['r2'][-1]:.4f}")
        if val_logs["z_sparsity"]:
            print(f"  L0 Sparsity: {val_logs['z_sparsity'][-1]:.2f}")
        if val_logs["dead_features"]:
            print(f"  Dead Features: {val_logs['dead_features'][-1] * 100:.1f}%")
    print("=" * 80 + "\n")

    # Return structured logs
    result = {"train": dict(train_logs)}
    if val_dataloader is not None and val_logs is not None:
        result["val"] = dict(val_logs)
    return result
