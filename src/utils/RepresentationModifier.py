from typing import Any, Dict  # noqa: N999

import torch
import torch.nn as nn
from einops import rearrange
from overcomplete.sae import TopKSAE
from torch import Tensor


class RepresentationModifier:
    def __init__(
        self,
        sae: TopKSAE,
        stats_dict: Dict[str, Any],
        # concepts_params_dict: Dict[str, Any],
        # features_number: int = 25,
        # influence_factor: float = 1.0,
        epsilon: float = 1e-8,
        device: str = "cuda",
        ignore_modification: str = "false",
    ):
        self.sae = sae.to(device)
        self.sae.eval()
        self.device = device
        self.stats_dict = stats_dict
        # self.influence_factor = influence_factor
        self.epsilon = epsilon
        # self.features_number = features_number
        self.timestep_idx = 0
        self.ignore_modification = ignore_modification
        self.concepts_to_unlearn_dict = {}

        # save means for 'all' stats for default feature selection
        all_timesteps = list(stats_dict["all"]["sums_per_timestep"].keys())
        all_timesteps.sort()

        sums_all = torch.stack(
            [stats_dict["all"]["sums_per_timestep"][t].to(torch.float32) for t in all_timesteps]
        )
        n_all = max(
            sum([stats_dict["all"]["counts_per_timestep"][t] for t in all_timesteps]),
            1,
        )
        self.mean_all_per_timestep = sums_all / n_all  # float32

        # === 1. Kluczowa poprawka: wymuś float32! ===
        # sums_true = torch.stack(
        #     [
        #         stats_dict["sums_true_per_timestep"][t].to(torch.float32)
        #         for t in stats_dict["timesteps"]
        #     ]
        # )
        # n_true = max(
        #     sum([stats_dict["counts_true_per_timestep"][t] for t in stats_dict["timesteps"]]),
        #     1,
        # )
        # sums_false = torch.stack(
        #     [
        #         stats_dict["sums_false_per_timestep"][t].to(torch.float32)
        #         for t in stats_dict["timesteps"]
        #     ]
        # )
        # n_false = max(
        #     sum([stats_dict["counts_false_per_timestep"][t] for t in stats_dict["timesteps"]]),
        #     1,
        # )

        # self.mean_true_per_timestep = sums_true / n_true  # float32
        # self.mean_false_per_timestep = sums_false / n_false  # float32
        # self.mean_all_per_timestep = (sums_true + sums_false) / (n_true + n_false)  # float32

        # # === 2. Reszta bez zmian ===
        # prob_true_per_timestep = self.mean_true_per_timestep / (
        #     self.mean_true_per_timestep.sum(dim=1, keepdim=True) + self.epsilon
        # )
        # prob_false_per_timestep = self.mean_false_per_timestep / (
        #     self.mean_false_per_timestep.sum(dim=1, keepdim=True) + self.epsilon
        # )
        # scores_per_timestep = prob_true_per_timestep - prob_false_per_timestep

        # self.top_indices = torch.topk(
        #     scores_per_timestep, k=features_number, dim=1
        # ).indices  # (timesteps, features_number)

        # # Zapisz jako float32!
        # self.mask_threshold = self.mean_all_per_timestep[
        #     torch.arange(len(stats_dict["timesteps"])).unsqueeze(1),
        #     self.top_indices,
        # ].to(self.device, dtype=torch.float32)  # (timesteps, features_number)
        # self.mean_true_top = self.mean_true_per_timestep[
        #     torch.arange(len(stats_dict["timesteps"])).unsqueeze(1),
        #     self.top_indices,
        # ].to(self.device, dtype=torch.float32)  # (timesteps, features_number)

        # print("RepresentationModifier initialized (float32 mode):")
        # print(f"  → Top {features_number} features: {self.top_indices}")
        # print(f"  → Dtype: {self.mean_true_top.dtype}")

    def add_concept_to_unlearn(
        self,
        concept_name: str,
        influence_factor: float,
        features_number: int,
    ):
        # 0. check if concept already added and if concept_name is valid
        if concept_name in self.concepts_to_unlearn_dict:
            print(f"Concept '{concept_name}' is already added to unlearn list.")
            return

        # 1. compute feauture selection for the concept (self.data_stats_dict['all'] contains overall stats)
        stats_concept = self.stats_dict[concept_name]
        stats_all = self.stats_dict["all"]

        # 2. timesteps must match, timesteps can be aquired from key of sums_per_timestep
        concept_timesteps = list(stats_concept["sums_per_timestep"].keys())
        concept_timesteps.sort()
        all_timesteps = list(stats_all["sums_per_timestep"].keys())
        all_timesteps.sort()
        if concept_timesteps != all_timesteps:
            raise ValueError(
                f"Timesteps for concept '{concept_name}' do not match overall stats timesteps."
            )

        timesteps = concept_timesteps

        sums_true = torch.stack(
            [stats_concept["sums_per_timestep"][t].to(torch.float32) for t in timesteps]
        )
        n_true = max(
            sum([stats_concept["counts_per_timestep"][t] for t in timesteps]),
            1,
        )
        sums_false = torch.stack(
            [
                stats_all["sums_per_timestep"][t].to(torch.float32)
                - stats_concept["sums_per_timestep"][t].to(torch.float32)
                for t in timesteps
            ]
        )
        n_false = max(
            sum([stats_all["counts_per_timestep"][t] for t in timesteps])
            - sum([stats_concept["counts_per_timestep"][t] for t in timesteps]),
            1,
        )

        mean_true_per_timestep = sums_true / n_true  # float32
        mean_false_per_timestep = sums_false / n_false  # float32
        prob_true_per_timestep = mean_true_per_timestep / (
            mean_true_per_timestep.sum(dim=1, keepdim=True) + self.epsilon
        )
        prob_false_per_timestep = mean_false_per_timestep / (
            mean_false_per_timestep.sum(dim=1, keepdim=True) + self.epsilon
        )
        scores_per_timestep = prob_true_per_timestep - prob_false_per_timestep
        top_indices = torch.topk(
            scores_per_timestep, k=features_number, dim=1
        ).indices  # (timesteps, features_number)
        # Zapisz jako float32!
        mean_true_top = mean_true_per_timestep[
            torch.arange(len(timesteps)).unsqueeze(1),
            top_indices,
        ].to(self.device, dtype=torch.float32)  # (timesteps, features_number
        # 2. store in the dict
        self.concepts_to_unlearn_dict[concept_name] = {
            "influence_factor": influence_factor,
            "features_number": features_number,
            "top_indices": top_indices,
            "mean_true_top": mean_true_top,
        }
        print(f"Concept '{concept_name}' added to unlearn list.")

    def remove_concept_to_unlearn(self, concept_name: str):
        if concept_name in self.concepts_to_unlearn_dict:
            del self.concepts_to_unlearn_dict[concept_name]
            print(f"Concept '{concept_name}' removed from unlearn list.")
        else:
            print(f"Concept '{concept_name}' not found in unlearn list.")

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
            # initialize modified codes as original
            codes_modified = codes_original.clone()
            for concept_name, concept_params in self.concepts_to_unlearn_dict.items():
                top_indices_t = concept_params["top_indices"][current_timestep]  # (k,)
                mean_true_top_t = concept_params["mean_true_top"][current_timestep]  # (k,)
                influence_factor = concept_params["influence_factor"]

                # === 5. Modyfikacja tylko wybranych cech ===
                codes_top = codes_modified[:, top_indices_t]  # [N_tokens, k]
                means_all_top_t = self.mean_all_per_timestep[
                    current_timestep, top_indices_t
                ]  # (k,)

                mask = (codes_top > means_all_top_t.unsqueeze(0)).float()
                new_values = -influence_factor * mean_true_top_t.unsqueeze(0) * codes_top

                codes_top_modified = codes_top * (1 - mask) + new_values * mask

                # Wstaw zmodyfikowane wartości z powrotem
                codes_modified[:, top_indices_t] = codes_top_modified

            # === 6. Dekoduj zmodyfikowane kody ===
            x_recon_modified = self.sae.decode(codes_modified)

            # === 7. Dodaj oryginalny reconstruction error (magia!) ===
            x_out_flat = x_recon_modified + reconstruction_error

            # === 8. Przywróć oryginalny kształt ===
            x_out = rearrange(x_out_flat, "(b t) d -> b t d", b=b, t=t)
            x_out = x_out.to(dtype=original_dtype)

            # current_timestep = self.timestep_idx
            # top_indices_t = self.top_indices[current_timestep]  # (k,)
            # mask_threshold_t = self.mask_threshold[current_timestep]  # (k,)
            # mean_true_top_t = self.mean_true_top[current_timestep]  # (k,)

            # # === 4. Enkoduj ponownie (można użyć codes_original – już mamy) ===
            # # _, codes = self.sae.encode(x_flat)  # [N_tokens, nb_concepts]
            # codes = codes_original.clone()
            # # (można też: codes = codes_original.clone() – szybciej)

            # # === 5. Modyfikacja tylko wybranych cech ===
            # codes_top = codes[:, top_indices_t]  # [N_tokens, k]

            # mask = (codes_top > mask_threshold_t).float()
            # new_values = -self.influence_factor * mean_true_top_t.unsqueeze(0) * codes_top

            # codes_top_modified = codes_top * (1 - mask) + new_values * mask

            # # Wstaw zmodyfikowane wartości z powrotem
            # codes_modified = codes.clone()
            # codes_modified[:, top_indices_t] = codes_top_modified

            # # === 6. Dekoduj zmodyfikowane kody ===
            # x_recon_modified = self.sae.decode(codes_modified)

            # # === 7. Dodaj oryginalny reconstruction error (magia!) ===
            # x_out_flat = x_recon_modified + reconstruction_error

            # # === 8. Przywróć oryginalny kształt ===
            # x_out = rearrange(x_out_flat, "(b t) d -> b t d", b=b, t=t)
            # x_out = x_out.to(dtype=original_dtype)

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
