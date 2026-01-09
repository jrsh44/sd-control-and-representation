from typing import Any, Dict  # noqa: N999

import torch
import torch.nn as nn
from einops import rearrange
from overcomplete.sae import TopKSAE


class RepresentationModifier:
    def __init__(
        self,
        sae: TopKSAE,
        stats_dict: Dict[str, Any],
        features_number: int = 25,
        influence_factor: float = 1.0,
        epsilon: float = 1e-8,
        device: str = "cuda",
        ignore_modification: str = "false",
    ):
        self.sae = sae.to(device)
        self.sae.eval()
        self.device = device
        self.influence_factor = influence_factor
        self.epsilon = epsilon
        self.features_number = features_number
        self.timestep_idx = 0
        self.ignore_modification = ignore_modification

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

    # def modification_hook(self, module: nn.Module, input: Any, output: Any) -> Any:
    #     if isinstance(output, tuple):
    #         x = output[0]
    #     else:
    #         x = output

    #     original_dtype = x.dtype
    #     x = x.float()  # SAE uses float32

    #     b, t, d = x.shape
    #     x_flat = rearrange(x, "b t d -> (b t) d")

    #     if self.ignore_modification_type == "no_sae":
    #         return output

    #     with torch.no_grad():
    #         _, codes = self.sae.encode(x_flat)  # codes is float32

    #         if self.ignore_modification_type == "raw_sae":
    #             x_recon = self.sae.decode(codes)
    #             x_out = rearrange(x_recon, "(b t) d -> b t d", b=b, t=t)
    #             x_out = x_out.to(dtype=original_dtype)
    #             return (x_out,) + output[1:] if isinstance(output, tuple) else x_out

    #         # Get timestep-specific indices and thresholds
    #         current_timestep = self.timestep_idx
    #         top_indices_t = self.top_indices[current_timestep]  # (features_number,)
    #         mask_threshold_t = self.mask_threshold[current_timestep]  # (features_number,)
    #         mean_true_top_t = self.mean_true_top[current_timestep]  # (features_number,)

    #         # Select only the top features for this timestep
    #         codes_top = codes[:, top_indices_t]  # (b*t, features_number)

    #         # Mask and calculations – all in float32
    #         mask = (codes_top > mask_threshold_t).float()
    #         new_values = -self.influence_factor * mean_true_top_t.unsqueeze(0) * codes_top

    #         codes_top_modified = codes_top * (1 - mask) + new_values * mask

    #         # Safe assignment
    #         codes_modified = codes.clone()
    #         codes_modified[:, top_indices_t] = codes_top_modified  # both float32 → OK

    #         x_recon = self.sae.decode(codes_modified)
    #         x_out = rearrange(x_recon, "(b t) d -> b t d", b=b, t=t)
    #         x_out = x_out.to(dtype=original_dtype)

    #     # Increment timestep for next call
    #     self.timestep_idx += 1

    #     return (x_out,) + output[1:] if isinstance(output, tuple) else x_out
    def modification_hook(self, module: nn.Module, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            x = output[0]
        else:
            x = output

        original_dtype = x.dtype
        x = x.float()  # SAE działa w float32

        b, t, d = x.shape
        x_flat = rearrange(x, "b t d -> (b t) d")  # [N_tokens, hidden_dim]

        # === 0. Wczesne wyjście dla trybów bez modyfikacji ===
        if self.ignore_modification == "true":
            return output

        with torch.no_grad():
            # === 1. Oryginalna rekonstrukcja SAE (przed jakąkolwiek modyfikacją) ===
            _, codes_original = self.sae.encode(x_flat)
            x_recon_original = self.sae.decode(codes_original)  # to SAE normalnie by zwróciło

            # === 2. Zachowaj reconstruction error (kluczowe!) ===
            reconstruction_error = x_flat - x_recon_original  # [N_tokens, d]

            # === 3. Pobierz parametry dla bieżącego timestepu ===
            current_timestep = self.timestep_idx
            top_indices_t = self.top_indices[current_timestep]  # (k,)
            mask_threshold_t = self.mask_threshold[current_timestep]  # (k,)
            mean_true_top_t = self.mean_true_top[current_timestep]  # (k,)

            # === 4. Enkoduj ponownie (można użyć codes_original – już mamy) ===
            # _, codes = self.sae.encode(x_flat)  # [N_tokens, nb_concepts]
            codes = codes_original.clone()
            # (można też: codes = codes_original.clone() – szybciej)

            # === 5. Modyfikacja tylko wybranych cech ===
            codes_top = codes[:, top_indices_t]  # [N_tokens, k]

            mask = (codes_top > mask_threshold_t).float()
            new_values = -self.influence_factor * mean_true_top_t.unsqueeze(0) * codes_top

            codes_top_modified = codes_top * (1 - mask) + new_values * mask

            # Wstaw zmodyfikowane wartości z powrotem
            codes_modified = codes.clone()
            codes_modified[:, top_indices_t] = codes_top_modified

            # === 6. Dekoduj zmodyfikowane kody ===
            x_recon_modified = self.sae.decode(codes_modified)

            # === 7. Dodaj oryginalny reconstruction error (magia!) ===
            x_out_flat = x_recon_modified + reconstruction_error

            # === 8. Przywróć oryginalny kształt ===
            x_out = rearrange(x_out_flat, "(b t) d -> b t d", b=b, t=t)
            x_out = x_out.to(dtype=original_dtype)

        # === 9. Zwiększ licznik timestepu ===
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
