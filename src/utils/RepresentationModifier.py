from typing import Any, Dict, List, Optional  # noqa: N999

import torch
import torch.nn as nn
from einops import rearrange
from overcomplete.sae import TopKSAE


class RepresentationModifier:
    """
    Klasa do modyfikacji aktywacji w warstwie SAE podczas generowania obrazu.
    Wykonuje unlearning top-N cech na podstawie różnicy znormalizowanych średnich aktywacji.
    """

    def __init__(
        self,
        sae: TopKSAE,
        means_true: torch.Tensor,  # [nb_concepts] – średnie aktywacje gdy koncept jest obecny
        means_false: torch.Tensor,  # [nb_concepts] – średnie gdy koncept nieobecny
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

        # Przenieś średnie na GPU
        self.means_true = means_true.to(device)
        self.means_false = means_false.to(device)

        # Oblicz scores: (P(true) - P(false)) ≈ różnica znormalizowanych średnich
        prob_true = self.means_true / (self.means_true.sum() + self.epsilon)
        prob_false = self.means_false / (self.means_false.sum() + self.epsilon)
        self.feature_scores = prob_true - prob_false

        # Wybierz top-N cech najbardziej specyficznych dla konceptu
        _, self.top_indices = torch.topk(self.feature_scores, k=features_number, largest=True)
        self.top_indices = self.top_indices.tolist()

        print("RepresentationModifier initialized:")
        print(f"  → Top {features_number} features to unlearn: {self.top_indices}")
        print(f"  → Influence factor: {influence_factor}")
        print(f"  → SAE device: {next(sae.parameters()).device}")

        # Wektor do odejmowania – tylko dla wybranych cech
        self.subtraction_vector = torch.zeros_like(self.means_false)
        self.subtraction_vector[self.top_indices] = self.means_false[self.top_indices]

    def modification_hook(self, module: nn.Module, input: Any, output: Any) -> Any:
        """
        Hook do podłączenia do warstwy U-Net.
        Modyfikuje tylko aktywacje w top-N cechach, gdy są powyżej średniej dla konceptu.
        """
        if isinstance(output, tuple):
            x = output[0]
        else:
            x = output

        original_dtype = x.dtype
        x = x.float()  # SAE działa w float32

        b, t, d = x.shape
        x_flat = rearrange(x, "b t d -> (b t) d")

        with torch.no_grad():
            _, codes = self.sae.encode(x_flat)  # [B*T, nb_concepts]

            # Tworzymy maskę: gdzie codes > means_true (czyli "wie o koncepcie")
            mask = (codes[:, self.top_indices] > self.means_true[self.top_indices]).float()

            # Odejmij tylko w tych miejscach: influence_factor * means_false
            subtraction = (
                mask * self.influence_factor * self.means_false[self.top_indices].unsqueeze(0)
            )
            codes[:, self.top_indices] -= subtraction

            # Dekoduj zmodyfikowane kody
            x_recon = self.sae.decode(codes)
            x_modified = rearrange(x_recon, "(b t) d -> b t d", b=b, t=t)
            x_modified = x_modified.to(dtype=original_dtype)

        if isinstance(output, tuple):
            return (x_modified,) + output[1:]
        return x_modified

    def attach_to(self, pipe, layer_path) -> "RepresentationModifier":
        """
        Podłącza hook do wybranej warstwy w U-Net.
        Zwraca self, żeby można było używać w with.
        """
        from src.models.sd_v1_5 import get_nested_module

        module = get_nested_module(pipe.unet, str(layer_path.value))
        self.handle = module.register_forward_hook(self.modification_hook)
        print(f"RepresentationModifier attached to: {layer_path.value}")
        return self

    def detach(self):
        """Usuwa hook."""
        if hasattr(self, "handle"):
            self.handle.remove()
            print("RepresentationModifier detached.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.detach()
