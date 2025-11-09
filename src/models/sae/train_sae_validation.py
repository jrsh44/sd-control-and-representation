"""
import time
from collections import defaultdict

import torch
from einops import rearrange

from ..metrics import l2, r2_score, l0_eps
from .trackers import DeadCodeTracker
"""

from einops import rearrange

from overcomplete.metrics import l2, r2_score, l0_eps
from overcomplete.sae.trackers import DeadCodeTracker

from collections import defaultdict
import time
import torch

def extract_input(batch):
    """
    Extracts the input data from a batch. Supports different batch types:
    - If the batch is a tuple or list, returns the first element.
    - If the batch is a dict, tries to return the value with key 'data' (or 'input').
    - Otherwise, assumes the batch is the input data.
    """
    if isinstance(batch, (tuple, list)):
        return batch[0]
    elif isinstance(batch, dict):
        return batch.get('data')
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
    float
        Reconstruction error.
    """
    if len(x.shape) == 4 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, 'n c w h -> (n w h) c')
    elif len(x.shape) == 3 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, 'n t c -> (n t) c')
    else:
        assert x.shape == x_hat.shape, "Input and output shapes must match."
        x_flatten = x

    r2 = r2_score(x_flatten, x_hat)

    return r2.item()


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
        logs['lr'].append(optimizer.param_groups[0]['lr'])
        logs['step_loss'].append(loss.item())

    if monitoring > 1:
        # store directly some z values
        # and the params / gradients norms
        logs['z'].append(z.detach()[::10])
        logs['z_l2'].append(l2(z).item())

        logs['dictionary_sparsity'].append(l0_eps(model.get_dictionary()).mean().item())
        logs['dictionary_norms'].append(l2(model.get_dictionary(), -1).mean().item())

        for name, param in model.named_parameters():
            if param.grad is not None:
                logs[f'params_norm_{name}'].append(l2(param).item())
                logs[f'params_grad_norm_{name}'].append(l2(param.grad).item())


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
    device : str, optional
        Device to run on.

    Returns
    -------
    dict
        Dictionary with keys:
            - 'train': logs for training
            - 'val': logs for validation (empty if val_dataloader is None)
    """
    train_logs = defaultdict(list)
    val_logs = defaultdict(list) if val_dataloader is not None else None

    # Initialize dead code tracker on first batch
    dead_tracker = None

    for epoch in range(nb_epochs):
        # ==================== TRAINING PHASE ====================
        model.train()
        start_time = time.time()
        epoch_loss = epoch_error = epoch_sparsity = 0.0
        batch_count = 0

        for batch in train_dataloader:
            batch_count += 1
            x = extract_input(batch).to(device, non_blocking=True)
            optimizer.zero_grad()

            z_pre, z, x_hat = model(x)
            loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())

            # Initialize dead tracker
            if dead_tracker is None:
                dead_tracker = DeadCodeTracker(z.shape[1], device)
            dead_tracker.update(z)

            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if monitoring:
                epoch_loss += loss.item()
                epoch_error += _compute_reconstruction_error(x, x_hat)
                epoch_sparsity += l0_eps(z, 0).sum().item()
                _log_metrics(monitoring, train_logs, model, z, loss, optimizer)

        # ==================== TRAINING METRICS ====================
        if monitoring and batch_count > 0:
            train_logs['avg_loss'].append(epoch_loss / batch_count)
            train_logs['r2'].append(epoch_error / batch_count)
            train_logs['z_sparsity'].append(epoch_sparsity / batch_count)
            train_logs['dead_features'].append(dead_tracker.get_dead_ratio())
            train_logs['time_epoch'].append(time.time() - start_time)

        # ==================== VALIDATION PHASE ====================
        if val_dataloader is not None:
            model.eval()
            val_loss = val_error = val_sparsity = 0.0
            val_batch_count = 0
            val_dead_tracker = DeadCodeTracker(z.shape[1], device) if dead_tracker is not None else None

            with torch.no_grad():
                for batch in val_dataloader:
                    val_batch_count += 1
                    x = extract_input(batch).to(device, non_blocking=True)
                    z_pre, z, x_hat = model(x)
                    loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())

                    val_loss += loss.item()
                    val_error += _compute_reconstruction_error(x, x_hat)
                    val_sparsity += l0_eps(z, 0).sum().item()

                    if val_dead_tracker is not None:
                        val_dead_tracker.update(z)

            if val_batch_count > 0:
                val_logs['avg_loss'].append(val_loss / val_batch_count)
                val_logs['r2'].append(val_error / val_batch_count)
                val_logs['z_sparsity'].append(val_sparsity / val_batch_count)
                val_logs['dead_features'].append(
                    val_dead_tracker.get_dead_ratio() if val_dead_tracker else 0.0
                )

        # ==================== LOGGING ====================
        if monitoring:
            train_str = (f"Epoch[{epoch+1}/{nb_epochs}] "
                         f"Train Loss: {train_logs['avg_loss'][-1]:.4f}, "
                         f"R2: {train_logs['r2'][-1]:.4f}, "
                         f"L0: {train_logs['z_sparsity'][-1]:.4f}, "
                         f"Dead: {train_logs['dead_features'][-1]*100:.1f}%")

            if val_dataloader is not None:
                val_str = (f" | Val Loss: {val_logs['avg_loss'][-1]:.4f}, "
                           f"R2: {val_logs['r2'][-1]:.4f}, "
                           f"L0: {val_logs['z_sparsity'][-1]:.4f}, "
                           f"Dead: {val_logs['dead_features'][-1]*100:.1f}%")
                print(train_str + val_str)
            else:
                print(train_str + f", Time: {train_logs['time_epoch'][-1]:.2f}s")

    # Return structured logs
    result = {'train': dict(train_logs)}
    if val_dataloader is not None:
        result['val'] = dict(val_logs)
    return result