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


def compute_avg_max_cosine_similarity(weight_matrix: torch.Tensor) -> float:
    """
    Compute the average maximum cosine similarity for a weight matrix.

    For each feature (row) in the weight matrix, computes its cosine similarity
    with all other features, finds the maximum similarity (excluding self),
    and returns the average of these maximum similarities.

    Args:
        weight_matrix (torch.Tensor): Weight matrix of shape (nb_features, feature_dim)

    Returns:
        float: Average maximum cosine similarity across all features
    """
    if weight_matrix.shape[0] > 10000:
        # Avoid OOM for very large dictionaries by doing it on CPU or in chunks if needed
        # For now, we try on device but detach to save graph memory
        weight_matrix = weight_matrix.detach()

    # Normalize weights to unit norm (L2 normalize each row)
    normalized = torch.nn.functional.normalize(weight_matrix, p=2, dim=1)

    # Compute absolute pairwise cosine similarities (nb_features x nb_features)
    # Using .mm instead of @ for clarity with tensors
    cos_sim = torch.mm(normalized, normalized.t()).abs()

    # Set diagonal to -inf to exclude self-similarity when finding max
    cos_sim.fill_diagonal_(-float("inf"))

    # For each feature, find max similarity with other features, then average
    avg_max_cos = cos_sim.max(dim=1)[0].mean().item()

    return avg_max_cos


def criterion_laux(x, x_hat, pre_codes, codes, dictionary):
    """
    Custom criterion function for Sparse Autoencoder (SAE) training.
    """
    n = x.shape[1]

    # Compute reconstruction MSE
    mse = torch.mean((x - x_hat) ** 2)

    # Hyperparameters
    alpha = 1 / 32  # Scaling factor for auxiliary loss
    k_aux = n // 2  # Number of least active features for auxiliary loss

    # Compute mean activation per feature across the batch
    feature_acts = codes.mean(dim=0)

    # Select indices of k_aux features with the lowest mean activations
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
    if monitoring == 0:
        return

    if monitoring > 0:
        logs["lr"].append(optimizer.param_groups[0]["lr"])
        logs["step_loss"].append(loss.item())

    if monitoring > 1:
        # store directly some z values
        logs["z"].append(z.detach()[::10])
        logs["z_l2"].append(l2(z).item())

        logs["dictionary_sparsity"].append(l0_eps(model.get_dictionary()).mean().item())
        logs["dictionary_norms"].append(l2(model.get_dictionary(), -1).mean().item())

        for name, param in model.named_parameters():
            if param.grad is not None:
                logs[f"params_norm_{name}"].append(l2(param).item())
                logs[f"params_grad_norm_{name}"].append(l2(param.grad).item())


def _train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    scheduler,
    dead_tracker,
    device,
    clip_grad,
    log_interval,
    monitoring,
    wandb_enabled,
    epoch,
    nb_epochs,
    train_logs,
):
    """
    Handles the training loop for a single epoch.
    Logs metrics to WandB every `log_interval` batches.
    """
    print(f"Epoch [{epoch + 1}/{nb_epochs}]")
    print("-" * 80)
    print("🏋️  Training phase...")

    model.train()
    start_time = time.time()

    epoch_loss_tensor = torch.tensor(0.0, device=device)
    epoch_error_tensor = torch.tensor(0.0, device=device)
    epoch_sparsity_tensor = torch.tensor(0.0, device=device)
    epoch_z_l2 = 0.0
    epoch_dict_sparsity = 0.0
    epoch_dict_norms = 0.0

    batch_count = 0
    metric_samples = 0
    iteration_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        batch_ready = time.time()
        batch_count += 1

        gpu_work_start = time.time()
        x = extract_input(batch).to(device, non_blocking=True)

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

        # Initialize dead tracker if needed
        if dead_tracker is None:
            dead_tracker = DeadCodeTracker(z.shape[1], device)
            print(f"   Initialized dead code tracker (tracking {z.shape[1]} features)")

        # Update dead tracker periodically
        if batch_idx % log_interval == 0 and dead_tracker is not None:
            dead_tracker.update(z.detach())

        if monitoring:
            epoch_loss_tensor += loss.detach()

            # Compute and log metrics every log_interval
            if batch_idx % log_interval == 0:
                metric_samples += 1
                with torch.no_grad():
                    z_detached = z.detach()
                    current_r2 = _compute_reconstruction_error(x, x_hat).item()
                    # l0_eps returns counts per feature, sum to get total, divide by batch size for average
                    current_l0 = l0_eps(z_detached, 0).sum().item() / x.shape[0]

                    epoch_error_tensor += current_r2
                    epoch_sparsity_tensor += current_l0

                    # Calculate Advanced Metrics for this batch/step
                    # 1. Cosine Similarity (Decoder)
                    decoder_weights = model.get_dictionary()
                    decoder_cos_sim = compute_avg_max_cosine_similarity(decoder_weights)
                    decoder_norm = torch.norm(decoder_weights, dim=1).mean().item()

                    # 2. Active Features (Current Batch Stats)
                    active_counts = {}
                    total_samples = z_detached.shape[0]
                    feature_active_frac = (z_detached > 0.01).float().sum(dim=0) / total_samples
                    for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
                        active_counts[f"active_features_{t}"] = (
                            (feature_active_frac > t).float().sum().item()
                        )

                    # Accumulators for epoch summary
                    epoch_z_l2 += l2(z_detached).item()
                    epoch_dict_sparsity += l0_eps(decoder_weights).mean().item()
                    epoch_dict_norms += l2(decoder_weights, dims=-1).mean().item()

                    # --- WANDB LOGGING PER BATCH/INTERVAL ---
                    if wandb_enabled and WANDB_AVAILABLE:
                        step_log = {
                            "train/loss": loss.item(),
                            "train/r2": current_r2,
                            "train/l0_sparsity": current_l0,
                            "train/decoder_avg_max_cos": decoder_cos_sim,
                            "train/decoder_mean_norm": decoder_norm,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1,
                            "batch": batch_idx,
                        }

                        if dead_tracker is not None:
                            step_log["train/dead_features_ratio"] = dead_tracker.get_dead_ratio()

                        # Add active feature counts
                        for k, v in active_counts.items():
                            step_log[f"train/{k}"] = v

                        # Basic Encoder Cosine Sim (if available)
                        if hasattr(model, "encoder"):
                            encoder_weights = None
                            if hasattr(model.encoder, "weight"):
                                encoder_weights = model.encoder.weight
                            elif hasattr(model.encoder, "linear") and hasattr(
                                model.encoder.linear, "weight"
                            ):
                                encoder_weights = model.encoder.linear.weight
                            elif hasattr(model.encoder, "final_block"):
                                try:
                                    encoder_weights = model.encoder.final_block[0].weight
                                except:
                                    pass

                            if encoder_weights is not None:
                                step_log["train/encoder_avg_max_cos"] = (
                                    compute_avg_max_cosine_similarity(encoder_weights)
                                )

                        wandb.log(step_log)

            # Log detailed internal list metrics (optional, for debugging/summary)
            if monitoring > 1 and batch_idx % log_interval == 0:
                _log_metrics(monitoring, train_logs, model, z.detach(), loss, optimizer)

        if device.type == "cuda":
            torch.cuda.synchronize()

        gpu_work_end = time.time()
        loading_time = batch_ready - iteration_start
        gpu_time = gpu_work_end - gpu_work_start
        total_batch_time = gpu_work_end - iteration_start
        iteration_start = time.time()

        # Track overall progress
        elapsed = time.time() - start_time
        avg_time_per_batch = elapsed / (batch_idx + 1)
        remaining_batches = len(dataloader) - (batch_idx + 1)
        eta_seconds = avg_time_per_batch * remaining_batches
        eta_minutes = int(eta_seconds // 60)
        eta_secs = int(eta_seconds % 60)

        progress_percent = int(((batch_idx + 1) / len(dataloader)) * 100)

        if batch_idx % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print(
                f"   [{progress_percent:3d}%] "
                f"Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Total: {total_batch_time:.3f}s "
                f"(Loading: {loading_time:.3f}s, GPU: {gpu_time:.3f}s) | "
                f"ETA: {eta_minutes}m {eta_secs:02d}s"
            )

    train_time = time.time() - start_time

    # Collect parameter and gradient norms at end of epoch
    params_norms = {}
    params_grad_norms = {}
    for name, param in model.named_parameters():
        params_norms[name] = l2(param).item()
        if param.grad is not None:
            params_grad_norms[name] = l2(param.grad).item()

    # Update summary logs (for print summary only)
    if monitoring and batch_count > 0:
        train_logs["avg_loss"].append((epoch_loss_tensor / batch_count).item())
        if metric_samples > 0:
            train_logs["r2"].append((epoch_error_tensor / metric_samples).item())
            train_logs["z_sparsity"].append((epoch_sparsity_tensor / metric_samples).item())
            train_logs["z_l2"].append(epoch_z_l2 / metric_samples)
            train_logs["dict_sparsity"].append(epoch_dict_sparsity / metric_samples)
            train_logs["dict_norms"].append(epoch_dict_norms / metric_samples)
        else:
            train_logs["z_l2"].append(0.0)
            train_logs["dict_sparsity"].append(0.0)
            train_logs["dict_norms"].append(0.0)

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

    return dead_tracker, train_time, params_norms, params_grad_norms


def _validate_one_epoch(model, dataloader, criterion, scaler, device, val_logs, log_interval):
    """
    Handles the validation loop for a single epoch.
    Calculates metrics for the entire validation set.
    """
    print("   🔍 Validation phase...")
    model.eval()

    val_loss_tensor = torch.tensor(0.0, device=device)
    val_error_tensor = torch.tensor(0.0, device=device)
    val_sparsity_tensor = torch.tensor(0.0, device=device)
    val_batch_count = 0
    val_active_features_accum = None
    val_total_samples = 0
    val_dead_tracker = None

    val_start_time = time.time()
    val_iteration_start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            val_batch_ready = time.time()
            val_batch_count += 1

            val_gpu_work_start = time.time()
            x = extract_input(batch).to(device, non_blocking=True)

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

            if val_dead_tracker is None:
                val_dead_tracker = DeadCodeTracker(z.shape[1], device)

            val_dead_tracker.update(z)

            if val_active_features_accum is None:
                val_active_features_accum = torch.zeros(z.shape[1], device="cpu")

            active_per_feature = (z > 0.01).float().sum(dim=0).cpu()
            val_active_features_accum += active_per_feature
            val_total_samples += z.shape[0]

            if device.type == "cuda":
                torch.cuda.synchronize()

            val_gpu_work_end = time.time()
            val_loading_time = val_batch_ready - val_iteration_start
            val_gpu_time = val_gpu_work_end - val_gpu_work_start
            val_total_batch_time = val_gpu_work_end - val_iteration_start
            val_iteration_start = time.time()

            val_elapsed = time.time() - val_start_time
            avg_time_per_batch = val_elapsed / (batch_idx + 1)
            remaining_batches = len(dataloader) - (batch_idx + 1)
            eta_seconds = avg_time_per_batch * remaining_batches
            eta_minutes = int(eta_seconds // 60)
            eta_secs = int(eta_seconds % 60)
            progress_percent = int(((batch_idx + 1) / len(dataloader)) * 100)

            if batch_idx % log_interval == 0:
                print(
                    f"   [{progress_percent:3d}%] "
                    f"Batch {batch_idx + 1}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Total: {val_total_batch_time:.3f}s "
                    f"(Loading: {val_loading_time:.3f}s, "
                    f"Calculations: {val_gpu_time:.3f}s) | "
                    f"ETA: {eta_minutes}m {eta_secs:02d}s"
                )

    if val_batch_count > 0 and val_logs is not None:
        val_logs["avg_loss"].append((val_loss_tensor / val_batch_count).item())
        val_logs["r2"].append((val_error_tensor / val_batch_count).item())
        val_logs["z_sparsity"].append((val_sparsity_tensor / val_batch_count).item())
        val_logs["dead_features"].append(
            val_dead_tracker.get_dead_ratio() if val_dead_tracker else 0.0
        )

        # --- Advanced Metrics Calculation (Validation) ---
        if val_total_samples > 0:
            feature_active_frac = val_active_features_accum / val_total_samples
            thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]
            for t in thresholds:
                count = (feature_active_frac > t).sum().item()
                val_logs[f"active_features_{t}"].append(count)
        else:
            for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
                val_logs[f"active_features_{t}"].append(0)

        decoder_weights = model.get_dictionary()
        decoder_cos_sim = compute_avg_max_cosine_similarity(decoder_weights)
        decoder_norm = torch.norm(decoder_weights, dim=1).mean().item()
        val_logs["decoder_avg_max_cos"].append(decoder_cos_sim)
        val_logs["decoder_mean_norm"].append(decoder_norm)

        encoder_weights = None
        if hasattr(model, "encoder"):
            if hasattr(model.encoder, "weight"):
                encoder_weights = model.encoder.weight
            elif hasattr(model.encoder, "linear") and hasattr(model.encoder.linear, "weight"):
                encoder_weights = model.encoder.linear.weight
            elif hasattr(model.encoder, "final_block"):
                try:
                    encoder_weights = model.encoder.final_block[0].weight
                except:
                    pass

        if encoder_weights is not None:
            encoder_cos_sim = compute_avg_max_cosine_similarity(encoder_weights)
            val_logs["encoder_avg_max_cos"].append(encoder_cos_sim)
        else:
            val_logs["encoder_avg_max_cos"].append(0.0)

        print(
            f"   📊 Val Loss: {val_logs['avg_loss'][-1]:.4f} | "
            f"R²: {val_logs['r2'][-1]:.4f} | "
            f"L0: {val_logs['z_sparsity'][-1]:.2f} | "
            f"Dead: {val_logs['dead_features'][-1] * 100:.1f}% | "
            f"CosSim: {val_logs['decoder_avg_max_cos'][-1]:.3f}"
        )


def _log_epoch_wandb(
    epoch, optimizer, params_norms, params_grad_norms, val_logs, wandb_enabled, device
):
    """
    Logs end-of-epoch metrics to WandB.
    This includes Validation metrics (once per epoch) and
    parameter/gradient norms (collected at end of epoch).
    """
    if not (wandb_enabled and WANDB_AVAILABLE):
        return

    log_dict = {
        "epoch": epoch + 1,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }

    # Add param/grad norms to epoch logs
    for name, val in params_norms.items():
        log_dict[f"train/params_norm_{name}"] = val
    for name, val in params_grad_norms.items():
        log_dict[f"train/params_grad_norm_{name}"] = val

    if val_logs is not None:
        if val_logs["avg_loss"]:
            log_dict.update(
                {
                    "val/loss": val_logs["avg_loss"][-1],
                    "val/r2": val_logs["r2"][-1],
                    "val/l0_sparsity": val_logs["z_sparsity"][-1],
                    "val/dead_features_ratio": val_logs["dead_features"][-1],
                    "val/decoder_avg_max_cos": val_logs["decoder_avg_max_cos"][-1],
                    "val/decoder_mean_norm": val_logs["decoder_mean_norm"][-1],
                }
            )

            if val_logs["encoder_avg_max_cos"]:
                log_dict["val/encoder_avg_max_cos"] = val_logs["encoder_avg_max_cos"][-1]

            for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
                k = f"active_features_{t}"
                if val_logs[k]:
                    log_dict[f"val/{k}"] = val_logs[k][-1]

    if device.type == "cuda":
        log_dict["gpu/memory_allocated_mb"] = torch.cuda.memory_allocated(device) / 1024**2
        log_dict["gpu/memory_reserved_mb"] = torch.cuda.memory_reserved(device) / 1024**2
        log_dict["gpu/max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2

    wandb.log(log_dict)


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
        (
            dead_tracker,
            train_time,
            params_norms,
            params_grad_norms,
        ) = _train_one_epoch(
            model,
            train_dataloader,
            criterion,
            optimizer,
            scaler,
            scheduler,
            dead_tracker,
            device,
            clip_grad,
            log_interval,
            monitoring,
            wandb_enabled,
            epoch,
            nb_epochs,
            train_logs,
        )

        if val_dataloader is not None:
            _validate_one_epoch(
                model, val_dataloader, criterion, scaler, device, val_logs, log_interval
            )

        _log_epoch_wandb(
            epoch,
            optimizer,
            params_norms,
            params_grad_norms,
            val_logs,
            wandb_enabled,
            device,
        )

        print()  # Empty line between epochs

    # ==================== TRAINING SUMMARY ====================
    print("\n" + "=" * 80)
    print("SAE TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total epochs completed: {nb_epochs}")
    print(f"Total training time: {sum(train_logs['time_epoch']):.2f}s")
    if train_logs["time_epoch"]:
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
        if val_logs["decoder_avg_max_cos"]:
            print(f"  Decoder Cos Sim: {val_logs['decoder_avg_max_cos'][-1]:.4f}")
    print("=" * 80 + "\n")

    # Return structured logs
    result = {"train": dict(train_logs)}
    if val_dataloader is not None and val_logs is not None:
        result["val"] = dict(val_logs)
    return result
