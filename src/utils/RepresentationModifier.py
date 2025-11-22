from typing import Any, Dict  # noqa: N999

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from overcomplete.sae import TopKSAE


class RepresentationModifier:
    # def __init__(
    #     self,
    #     sae: TopKSAE,
    #     stats_dict: Dict[str, Any],
    #     features_number: int = 25,
    #     influence_factor: float = 1.0,
    #     epsilon: float = 1e-8,
    #     device: str = "cuda",
    # ):
    #     # stats_dict = {
    #     #     "sums_true_per_timestep": {0: Tensor, 1: Tensor, ..., 49: Tensor},
    #     #     "counts_true_per_timestep": {0: 1000, 1: 1000, ..., 49: 1000},
    #     #     "sums_false_per_timestep": {0: 5000, 1: 5000, ..., 49: 5000},
    #     #     "counts_false_per_timestep": {0: 5000, 1: 5000, ..., 49: 5000},
    #     #     "timesteps": [0, 1, 2, ..., 49]
    #     # }

    #     self.sae = sae.to(device)
    #     self.sae.eval()
    #     self.device = device
    #     self.influence_factor = influence_factor
    #     self.epsilon = epsilon
    #     self.features_number = features_number

    #     # === 1. Kluczowa poprawka: wymuś float32! ===
    #     sums_true = torch.stack(
    #         [
    #             stats_dict["sums_true_per_timestep"][t].to(torch.float32)
    #             for t in stats_dict["timesteps"]
    #         ]
    #     ).sum(dim=0)
    #     n_true = max(
    #         sum([stats_dict["counts_true_per_timestep"][t] for t in stats_dict["timesteps"]]),
    #         1,
    #     )
    #     sums_false = torch.stack(
    #         [
    #             stats_dict["sums_false_per_timestep"][t].to(torch.float32)
    #             for t in stats_dict["timesteps"]
    #         ]
    #     ).sum(dim=0)
    #     n_false = max(
    #         sum([stats_dict["counts_false_per_timestep"][t] for t in stats_dict["timesteps"]]),
    #         1,
    #     )

    #     self.mean_true = sums_true / n_true
    #     self.mean_false = sums_false / n_false
    #     self.mean_all = (sums_true + sums_false) / (n_true + n_false)

    #     # === 2. Reszta bez zmian ===
    #     prob_true = self.mean_true / (self.mean_true.sum() + self.epsilon)
    #     prob_false = self.mean_false / (self.mean_false.sum() + self.epsilon)
    #     scores = prob_true - prob_false

    #     _, top_indices = torch.topk(scores, k=features_number, largest=True)
    #     self.top_indices = top_indices.tolist()

    #     # Zapisz jako float32!
    #     self.mask_threshold = self.mean_all[self.top_indices].to(self.device, dtype=torch.float32)
    #     self.mean_true_top = self.mean_true[self.top_indices].to(self.device, dtype=torch.float32)

    #     print("RepresentationModifier initialized (float32 mode):")
    #     print(f"  → Top {features_number} features: {self.top_indices}")
    #     print(f"  → Dtype: {self.mean_true_top.dtype}")

    # def modification_hook(self, module: nn.Module, input: Any, output: Any) -> Any:
    #     if isinstance(output, tuple):
    #         x = output[0]
    #     else:
    #         x = output

    #     original_dtype = x.dtype
    #     x = x.float()  # SAE używa float32

    #     b, t, d = x.shape
    #     x_flat = rearrange(x, "b t d -> (b t) d")

    #     with torch.no_grad():
    #         _, codes = self.sae.encode(x_flat)  # codes jest float32

    #         codes_top = codes[:, self.top_indices]  # float32

    #         # Maska i obliczenia – wszystko w float32
    #         mask = (codes_top > self.mask_threshold).float()
    #         new_values = -self.influence_factor * self.mean_true_top.unsqueeze(0) * codes_top

    #         codes_top_modified = codes_top * (1 - mask) + new_values * mask

    #         # Teraz bezpieczne przypisanie
    #         codes_modified = codes.clone()
    #         codes_modified[:, self.top_indices] = codes_top_modified  # oba float32 → OK

    #         x_recon = self.sae.decode(codes_modified)
    #         x_out = rearrange(x_recon, "(b t) d -> b t d", b=b, t=t)
    #         x_out = x_out.to(dtype=original_dtype)

    #     return (x_out,) + output[1:] if isinstance(output, tuple) else x_out
    def __init__(
        self,
        sae: TopKSAE,
        stats_dict: Dict[str, Any],
        features_number: int = 25,
        influence_factor: float = 1.0,
        epsilon: float = 1e-8,
        device: str = "cuda",
    ):
        self.sae = sae.to(device)
        self.sae.eval()
        self.device = device
        self.influence_factor = influence_factor
        self.epsilon = epsilon
        self.features_number = features_number
        self.timestep_idx = 0

        # === 1. Kluczowa poprawka: wymuś float32! ===
        sums_true = torch.stack(
            [
                stats_dict["sums_true_per_timestep"][t].to(torch.float32)
                for t in stats_dict["timesteps"]
            ]
        )
        n_true = max(
            sum([stats_dict["counts_true_per_timestep"][t] for t in stats_dict["timesteps"]]),
            1,
        )
        sums_false = torch.stack(
            [
                stats_dict["sums_false_per_timestep"][t].to(torch.float32)
                for t in stats_dict["timesteps"]
            ]
        )
        n_false = max(
            sum([stats_dict["counts_false_per_timestep"][t] for t in stats_dict["timesteps"]]),
            1,
        )

        self.mean_true_per_timestep = sums_true / n_true  # float32
        self.mean_false_per_timestep = sums_false / n_false  # float32
        self.mean_all_per_timestep = (sums_true + sums_false) / (n_true + n_false)  # float32

        # === 2. Reszta bez zmian ===
        prob_true_per_timestep = self.mean_true_per_timestep / (
            self.mean_true_per_timestep.sum(dim=1, keepdim=True) + self.epsilon
        )
        prob_false_per_timestep = self.mean_false_per_timestep / (
            self.mean_false_per_timestep.sum(dim=1, keepdim=True) + self.epsilon
        )
        scores_per_timestep = prob_true_per_timestep - prob_false_per_timestep

        self.top_indices = torch.topk(
            scores_per_timestep, k=features_number, dim=1
        ).indices  # (timesteps, features_number)

        # Zapisz jako float32!
        self.mask_threshold = self.mean_all_per_timestep[
            torch.arange(len(stats_dict["timesteps"])).unsqueeze(1),
            self.top_indices,
        ].to(self.device, dtype=torch.float32)  # (timesteps, features_number)
        self.mean_true_top = self.mean_true_per_timestep[
            torch.arange(len(stats_dict["timesteps"])).unsqueeze(1),
            self.top_indices,
        ].to(self.device, dtype=torch.float32)  # (timesteps, features_number)

        print("RepresentationModifier initialized (float32 mode):")
        print(f"  → Top {features_number} features: {self.top_indices}")
        print(f"  → Dtype: {self.mean_true_top.dtype}")

    def modification_hook(self, module: nn.Module, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            x = output[0]
        else:
            x = output

        original_dtype = x.dtype
        x = x.float()  # SAE uses float32

        b, t, d = x.shape
        x_flat = rearrange(x, "b t d -> (b t) d")

        with torch.no_grad():
            _, codes = self.sae.encode(x_flat)  # codes is float32

            # Get timestep-specific indices and thresholds
            current_timestep = self.timestep_idx
            top_indices_t = self.top_indices[current_timestep]  # (features_number,)
            mask_threshold_t = self.mask_threshold[current_timestep]  # (features_number,)
            mean_true_top_t = self.mean_true_top[current_timestep]  # (features_number,)

            # Select only the top features for this timestep
            codes_top = codes[:, top_indices_t]  # (b*t, features_number)

            # Mask and calculations – all in float32
            mask = (codes_top > mask_threshold_t).float()
            new_values = -self.influence_factor * mean_true_top_t.unsqueeze(0) * codes_top

            codes_top_modified = codes_top * (1 - mask) + new_values * mask

            # Safe assignment
            codes_modified = codes.clone()
            codes_modified[:, top_indices_t] = codes_top_modified  # both float32 → OK

            x_recon = self.sae.decode(codes_modified)
            x_out = rearrange(x_recon, "(b t) d -> b t d", b=b, t=t)
            x_out = x_out.to(dtype=original_dtype)

        # Increment timestep for next call
        self.timestep_idx += 1

        return (x_out,) + output[1:] if isinstance(output, tuple) else x_out

    def reset_timestep(self):
        self.timestep_idx = 0

    def attach_to(self, pipe, layer_path) -> "RepresentationModifier":
        from src.models.sd_v1_5.hooks import get_nested_module

        module = get_nested_module(pipe, str(layer_path.value))
        self.handle = module.register_forward_hook(self.modification_hook)
        print(f"RepresentationModifier attached to: {layer_path.value}")
        return self

    def detach(self):
        if hasattr(self, "handle"):
            self.handle.remove()
            print("RepresentationModifier detached.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.detach()
