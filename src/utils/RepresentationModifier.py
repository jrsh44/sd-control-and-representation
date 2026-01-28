from typing import Any, Dict  # noqa: N999

import torch
import torch.nn as nn
from einops import rearrange
from overcomplete.sae import TopKSAE


class RepresentationModifier:
    def __init__(
        self,
        sae: TopKSAE | None,
        stats_dict: Dict[str, Any],
        epsilon: float = 1e-8,
        device: str = "cuda",
        ignore_modification: str = "false",
        max_concepts_number: int = 32,
    ):
        if sae is None:
            print(
                "Warning: SAE model is None. RepresentationModifier will not modify anything, only calculate scores."  # noqa: E501
            )
        else:
            self.sae = sae.to(device)
            self.sae.eval()
        self.device = device
        self.stats_dict = stats_dict
        self.epsilon = epsilon
        self.timestep_idx = 0
        self.ignore_modification = ignore_modification
        self.max_concepts_number = max_concepts_number
        self.concepts_to_unlearn_dict = {}

        all_timesteps = list(stats_dict["all"]["sums_per_timestep"].keys())
        all_timesteps.sort()

        sums_all = torch.stack(
            [stats_dict["all"]["sums_per_timestep"][t].to(torch.float32) for t in all_timesteps]
        )

        n_all = torch.tensor(
            [stats_dict["all"]["counts_per_timestep"][t] for t in all_timesteps],
            dtype=torch.float32,
        ).to(self.device)

        self.mean_all_per_timestep = sums_all / n_all.unsqueeze(1)

    def add_concept_to_unlearn(
        self,
        concept_name: str,
        influence_factor: float,
        features_number: int,
        per_timestep: bool = True,
    ):
        if concept_name in self.concepts_to_unlearn_dict:
            print(f"Concept '{concept_name}' is already added to unlearn list.")
            return

        stats_concept = self.stats_dict[concept_name]
        stats_all = self.stats_dict["all"]

        concept_timesteps = list(stats_concept["sums_per_timestep"].keys())
        concept_timesteps.sort()
        all_timesteps = list(stats_all["sums_per_timestep"].keys())
        all_timesteps.sort()
        if concept_timesteps != all_timesteps:
            raise ValueError(
                f"Timesteps for concept '{concept_name}' do not match overall stats timesteps."
            )

        timesteps = concept_timesteps

        if per_timestep:
            sums_true = torch.stack(
                [stats_concept["sums_per_timestep"][t].to(torch.float32) for t in timesteps]
            )
            n_true = torch.tensor(
                [stats_concept["counts_per_timestep"][t] for t in timesteps],
                dtype=torch.float32,
            ).to(self.device)

            sums_false = torch.stack(
                [
                    stats_all["sums_per_timestep"][t].to(torch.float32)
                    - stats_concept["sums_per_timestep"][t].to(torch.float32)
                    for t in timesteps
                ]
            )

            n_false = torch.tensor(
                [
                    stats_all["counts_per_timestep"][t] - stats_concept["counts_per_timestep"][t]
                    for t in timesteps
                ],
                dtype=torch.float32,
            ).to(self.device)

            mean_true_per_timestep = sums_true / n_true.unsqueeze(1)
            mean_false_per_timestep = sums_false / n_false.unsqueeze(1)

            prob_true_per_timestep = mean_true_per_timestep / (
                mean_true_per_timestep.sum(dim=1, keepdim=True) + self.epsilon
            )
            prob_false_per_timestep = mean_false_per_timestep / (
                mean_false_per_timestep.sum(dim=1, keepdim=True) + self.epsilon
            )
            scores_per_timestep = prob_true_per_timestep - prob_false_per_timestep
            top_indices = torch.topk(scores_per_timestep, k=self.max_concepts_number, dim=1).indices

            mean_true_top = mean_true_per_timestep[
                torch.arange(len(timesteps)).unsqueeze(1),
                top_indices,
            ].to(self.device, dtype=torch.float32)
        else:
            sum_true_aggregated = sum(
                [stats_concept["sums_per_timestep"][t].to(torch.float32) for t in timesteps]
            )
            n_true = max(
                sum([stats_concept["counts_per_timestep"][t] for t in timesteps]),
                1,
            )
            sum_false_aggregated = sum(
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

            mean_true_aggregated = sum_true_aggregated / n_true
            mean_false_aggregated = sum_false_aggregated / n_false
            prob_true_aggregated = mean_true_aggregated / (
                mean_true_aggregated.sum() + self.epsilon
            )
            prob_false_aggregated = mean_false_aggregated / (
                mean_false_aggregated.sum() + self.epsilon
            )
            scores_aggregated = prob_true_aggregated - prob_false_aggregated

            top_indices_1d = torch.topk(
                scores_aggregated, k=self.max_concepts_number, dim=0
            ).indices

            top_indices = top_indices_1d.unsqueeze(0)
            mean_true_top = (
                mean_true_aggregated[top_indices_1d]
                .unsqueeze(0)
                .to(self.device, dtype=torch.float32)
            )

        self.concepts_to_unlearn_dict[concept_name] = {
            "influence_factor": influence_factor,
            "features_number": features_number,
            "top_indices": top_indices,
            "mean_true_top": mean_true_top,
            "per_timestep": per_timestep,
        }
        mode_str = "per-timestep" if per_timestep else "all-timesteps"
        print(f"Concept '{concept_name}' added to unlearn list (mode: {mode_str}).")

    def calculate_scores_for_concept(
        self,
        concept_name: str,
        per_timestep: bool = True,
    ):
        if concept_name in self.concepts_to_unlearn_dict:
            print(f"Concept '{concept_name}' is already added to unlearn list.")
            return

        stats_concept = self.stats_dict[concept_name]
        stats_all = self.stats_dict["all"]

        concept_timesteps = list(stats_concept["sums_per_timestep"].keys())
        concept_timesteps.sort()
        all_timesteps = list(stats_all["sums_per_timestep"].keys())
        all_timesteps.sort()
        if concept_timesteps != all_timesteps:
            raise ValueError(
                f"Timesteps for concept '{concept_name}' do not match overall stats timesteps."
            )

        timesteps = concept_timesteps

        if per_timestep:
            sums_true = torch.stack(
                [stats_concept["sums_per_timestep"][t].to(torch.float32) for t in timesteps]
            )
            n_true = torch.tensor(
                [stats_concept["counts_per_timestep"][t] for t in timesteps],
                dtype=torch.float32,
            ).to(self.device)
            sums_false = torch.stack(
                [
                    stats_all["sums_per_timestep"][t].to(torch.float32)
                    - stats_concept["sums_per_timestep"][t].to(torch.float32)
                    for t in timesteps
                ]
            )
            n_false = torch.tensor(
                [
                    stats_all["counts_per_timestep"][t] - stats_concept["counts_per_timestep"][t]
                    for t in timesteps
                ],
                dtype=torch.float32,
            ).to(self.device)

            mean_true_per_timestep = sums_true / n_true.unsqueeze(1)
            mean_false_per_timestep = sums_false / n_false.unsqueeze(1)

            prob_true_per_timestep = mean_true_per_timestep / (
                mean_true_per_timestep.sum(dim=1, keepdim=True) + self.epsilon
            )
            prob_false_per_timestep = mean_false_per_timestep / (
                mean_false_per_timestep.sum(dim=1, keepdim=True) + self.epsilon
            )
            scores = prob_true_per_timestep - prob_false_per_timestep

        else:
            sum_true_aggregated = sum(
                [stats_concept["sums_per_timestep"][t].to(torch.float32) for t in timesteps]
            )
            n_true = max(
                sum([stats_concept["counts_per_timestep"][t] for t in timesteps]),
                1,
            )
            sum_false_aggregated = sum(
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

            mean_true_aggregated = sum_true_aggregated / n_true
            mean_false_aggregated = sum_false_aggregated / n_false
            prob_true_aggregated = mean_true_aggregated / (
                mean_true_aggregated.sum() + self.epsilon
            )
            prob_false_aggregated = mean_false_aggregated / (
                mean_false_aggregated.sum() + self.epsilon
            )
            scores = prob_true_aggregated - prob_false_aggregated

        return scores

    def set_number_of_features_for_concept(
        self,
        concept_name: str,
        features_number: int,
    ):
        if concept_name not in self.concepts_to_unlearn_dict:
            print(f"Concept '{concept_name}' not found in unlearn list.")
            raise ValueError(f"Concept '{concept_name}' not found in unlearn list.")

        if features_number > self.max_concepts_number:
            print(
                f"Requested features_number {features_number} exceeds max_concepts_number {self.max_concepts_number}."  # noqa: E501
            )
            raise ValueError(
                f"Requested features_number {features_number} exceeds max_concepts_number {self.max_concepts_number}."  # noqa: E501
            )

        self.concepts_to_unlearn_dict[concept_name]["features_number"] = features_number
        print(f"Number of features for concept '{concept_name}' set to {features_number}.")

    def set_influence_factor_for_concept(
        self,
        concept_name: str,
        influence_factor: float,
    ):
        if concept_name not in self.concepts_to_unlearn_dict:
            print(f"Concept '{concept_name}' not found in unlearn list.")
            return

        self.concepts_to_unlearn_dict[concept_name]["influence_factor"] = influence_factor
        print(f"Influence factor for concept '{concept_name}' set to {influence_factor}.")

    def set_ignore_modification(self):
        self.ignore_modification = "true"
        print("Modifications will be ignored.")

    def unset_ignore_modification(self):
        self.ignore_modification = "false"
        print("Modifications will be applied.")

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
        x = x.float()

        b, t, d = x.shape
        x_flat = rearrange(x, "b t d -> (b t) d")

        if self.ignore_modification == "true":
            return output

        if self.timestep_idx == 0:
            self.timestep_idx += 1
            return output

        with torch.no_grad():
            _, codes_original = self.sae.encode(x_flat)
            x_recon_original = self.sae.decode(codes_original)

            reconstruction_error = x_flat - x_recon_original

            current_timestep = self.timestep_idx - 1
            codes_modified = codes_original.clone()
            for concept_name, concept_params in self.concepts_to_unlearn_dict.items():
                if concept_params["per_timestep"]:
                    max_timestep_idx = concept_params["top_indices"].shape[0] - 1
                    timestep_idx = min(current_timestep, max_timestep_idx)
                else:
                    timestep_idx = 0
                top_indices_t = concept_params["top_indices"][timestep_idx][
                    : concept_params["features_number"]
                ]
                mean_true_top_t = concept_params["mean_true_top"][timestep_idx][
                    : concept_params["features_number"]
                ]
                influence_factor = concept_params["influence_factor"]

                codes_top = codes_modified[:, top_indices_t]
                if concept_params["per_timestep"]:
                    means_all_top_t = self.mean_all_per_timestep[timestep_idx, top_indices_t]
                else:
                    means_all_top_t = self.mean_all_per_timestep[:, top_indices_t].mean(dim=0)

                mask = (codes_top > means_all_top_t.unsqueeze(0)).float()
                new_values = -influence_factor * mean_true_top_t.unsqueeze(0) * codes_top

                codes_top_modified = codes_top * (1 - mask) + new_values * mask

                codes_modified[:, top_indices_t] = codes_top_modified

            x_recon_modified = self.sae.decode(codes_modified)

            x_out_flat = x_recon_modified + reconstruction_error

            x_out = rearrange(x_out_flat, "(b t) d -> b t d", b=b, t=t)
            x_out = x_out.to(dtype=original_dtype)

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
