import time

import torch

try:
    import wandb  # noqa: F401

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@torch.no_grad()
def get_codes(sae, x, device):
    _, z, _ = sae(x.to(device))
    return z


def concept_filtering_function(
    concept_name: str,
    concept_value: str,
    negate: bool = False,
) -> callable:
    if negate:
        return lambda x: x[concept_name] != concept_value
    else:
        return lambda x: x[concept_name] == concept_value


@torch.no_grad()
def compute_sums(
    loader, sae, device, nb_concepts, log_every: int = 50, phase_name: str = "compute_sums"
):
    """
    Oblicza średnią aktywacji koncepcji i loguje do wandb czas przetwarzania batchy.

    Args:
        loader: DataLoader
        sae: model SAE
        device: 'cuda' lub 'cpu'
        nb_concepts: liczba koncepcji w SAE
        log_every: co ile batchy logować statystyki
        phase_name: nazwa fazy (np. "concept_true" lub "concept_false")
    """
    total_sum = torch.zeros(nb_concepts, dtype=torch.float64, device="cpu")

    batch_times = []
    total_samples = 0
    start_time = time.time()

    print(f"Starting {phase_name}...")

    for i, batch in enumerate(loader):
        batch_start = time.time()

        codes = get_codes(sae, batch, device)
        codes_cpu = codes.to("cpu", dtype=torch.float64)
        total_sum += codes_cpu.sum(dim=0)

        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
        total_samples += batch.size(0)

        if (i + 1) % log_every == 0 or (i + 1) == len(loader):
            elapsed = time.time() - start_time
            avg_time = sum(batch_times[-log_every:]) / len(batch_times[-log_every:])
            speed_samples = total_samples / elapsed
            speed_batches = (i + 1) / elapsed

            print(
                f"  [{phase_name}] Batch {i + 1}/{len(loader)} | "
                f"Avg batch time: {avg_time * 1000:.1f}ms | "
                f"Speed: {speed_samples:,.0f} samples/s"
            )

            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(
                    {
                        f"{phase_name}/batch_time_ms": avg_time * 1000,
                        f"{phase_name}/min_batch_time_ms": min(batch_times) * 1000,
                        f"{phase_name}/max_batch_time_ms": max(batch_times) * 1000,
                        f"{phase_name}/throughput_samples_per_s": speed_samples,
                        f"{phase_name}/throughput_batches_per_s": speed_batches,
                        f"{phase_name}/processed_batches": i + 1,
                        f"{phase_name}/processed_samples": total_samples,
                    },
                    step=i,
                    commit=True,
                )

    total_time = time.time() - start_time
    print(
        f"{phase_name} finished in {total_time:.1f}s "
        f"({total_samples:,} samples, {total_samples / total_time:,.0f} samples/s)"
    )

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(
            {
                f"{phase_name}/total_time_s": total_time,
                f"{phase_name}/final_throughput_samples_per_s": total_samples / total_time,
            }
        )

    return total_sum


@torch.no_grad()
def compute_sums_per_timestep(
    loader, sae, device, nb_concepts, log_every: int = 50, phase_name: str = "compute_sums"
):
    """
    Computes sum of activations PER TIMESTEP.
    Optimized for large datasets with explicit memory management.

    Returns:
        sums_per_timestep: dict[int, torch.Tensor] - sum of activations for each timestep
        counts_per_timestep: dict[int, int] - number of samples for each timestep
    """
    sums_per_timestep = {}
    counts_per_timestep = {}

    batch_times = []
    total_samples = 0
    start_time = time.time()
    is_cuda = str(device) == "cuda"

    print(f"Starting {phase_name} (per-timestep)...")

    for i, (batch, timesteps) in enumerate(loader):
        batch_start = time.time()

        codes = get_codes(sae, batch, device)

        codes_cpu = codes.to("cpu", dtype=torch.float32)
        del codes

        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, dtype=torch.long)
        else:
            timesteps = timesteps.cpu()

        unique_timesteps = torch.unique(timesteps)
        for t in unique_timesteps:
            t_val = int(t.item())
            mask = timesteps == t
            codes_t = codes_cpu[mask]

            if t_val not in sums_per_timestep.keys():
                sums_per_timestep[t_val] = torch.zeros(nb_concepts, dtype=torch.float64)
                counts_per_timestep[t_val] = 0

            sums_per_timestep[t_val] += codes_t.sum(dim=0, dtype=torch.float64)
            counts_per_timestep[t_val] += codes_t.shape[0]
            del codes_t

        if is_cuda and (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
        total_samples += batch.size(0)

        if (i + 1) % log_every == 0 or (i + 1) == len(loader):
            elapsed = time.time() - start_time
            avg_time = sum(batch_times[-log_every:]) / len(batch_times[-log_every:])
            speed_samples = total_samples / elapsed
            speed_batches = (i + 1) / elapsed

            mem_info = ""
            if is_cuda:
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                mem_info = f" | GPU: {mem_allocated:.1f}GB/{mem_reserved:.1f}GB"

            print(
                f"  [{phase_name}] Batch {i + 1}/{len(loader)} | "
                f"Avg batch time: {avg_time * 1000:.1f}ms | "
                f"Speed: {speed_samples:,.0f} samples/s | "
                f"Timesteps: {len(sums_per_timestep)}{mem_info}"
            )

            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(
                    {
                        f"{phase_name}/batch_time_ms": avg_time * 1000,
                        f"{phase_name}/min_batch_time_ms": min(batch_times) * 1000,
                        f"{phase_name}/max_batch_time_ms": max(batch_times) * 1000,
                        f"{phase_name}/throughput_samples_per_s": speed_samples,
                        f"{phase_name}/throughput_batches_per_s": speed_batches,
                        f"{phase_name}/processed_batches": i + 1,
                        f"{phase_name}/processed_samples": total_samples,
                        f"{phase_name}/unique_timesteps": len(sums_per_timestep),
                        f"{phase_name}/elapsed_time_s": elapsed,
                    },
                    step=i,
                    commit=True,
                )

    total_time = time.time() - start_time
    print(
        f"{phase_name} finished in {total_time:.1f}s "
        f"({total_samples:,} samples, {len(sums_per_timestep)} timesteps)"
    )

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(
            {
                f"total_{phase_name}/total_time_s": total_time,
                f"total_{phase_name}/final_throughput_samples_per_s": total_samples / total_time,
                f"total_{phase_name}/total_samples": total_samples,
                f"total_{phase_name}/total_timesteps": len(sums_per_timestep),
            }
        )

    return sums_per_timestep, counts_per_timestep
