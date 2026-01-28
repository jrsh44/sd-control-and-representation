"""CLIP-based similarity metrics for evaluating generated images."""

from typing import TYPE_CHECKING

import torch
from PIL import Image
from torchmetrics.multimodal import CLIPScore
from torchvision import transforms

if TYPE_CHECKING:
    from core.state import DashboardState


_clip_model = None
_clip_device = None


def get_clip_model(device: str = "cpu", state: "DashboardState | None" = None) -> CLIPScore:
    """Get or initialize the CLIP model.

    Uses the state's clip_model if available, otherwise falls back to module cache.

    Args:
        device: Device to load model on ('cpu' or 'cuda').
        state: Optional dashboard state that may have pre-loaded CLIP model.

    Returns:
        CLIPScore metric instance.
    """
    global _clip_model, _clip_device

    if state is not None and state.clip_model is not None:
        if hasattr(state, "clip_device") and state.clip_device != device:
            state.clip_model = state.clip_model.to(device)
            state.clip_device = device
        return state.clip_model

    if _clip_model is None or _clip_device != device:
        _clip_model = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
        _clip_model = _clip_model.to(device)
        _clip_device = device

    return _clip_model


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor format expected by CLIPScore.

    Args:
        image: PIL Image to convert.

    Returns:
        Tensor of shape [C, H, W] with values in [0, 255].
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = transforms.ToTensor()
    tensor = transform(image)

    tensor = (tensor * 255).to(torch.uint8)

    return tensor


def calculate_clip_scores(
    prompt: str,
    original_image: Image.Image | None,
    intervention_image: Image.Image | None,
    device: str = "cpu",
    state: "DashboardState | None" = None,
) -> dict:
    """Calculate CLIP scores for prompt-image and image-image comparisons.

    Args:
        prompt: Text prompt used for generation.
        original_image: Original generated image (PIL).
        intervention_image: Image generated with intervention (PIL).
        device: Device for computation ('cpu' or 'cuda').
        state: Optional dashboard state with pre-loaded CLIP model.

    Returns:
        Dictionary with keys: 'clip_image_image', 'clip_prompt_original',
        'clip_prompt_intervention', or None if images are missing.
    """

    results = {
        "prompt_original": None,
        "prompt_intervention": None,
        "image_similarity": None,
    }

    if not prompt:
        return results

    try:
        metric = get_clip_model(device, state)

        if original_image is not None:
            try:
                orig_tensor = pil_to_tensor(original_image)
                metric.reset()
                metric.update(orig_tensor, prompt)
                score = metric.compute()
                results["prompt_original"] = float(score.item())
            except Exception as e:
                print(f"[CLIP] Error computing prompt vs original: {e}")

        if intervention_image is not None:
            try:
                interv_tensor = pil_to_tensor(intervention_image)
                metric.reset()
                metric.update(interv_tensor, prompt)
                score = metric.compute()
                results["prompt_intervention"] = float(score.item())
            except Exception as e:
                print(f"[CLIP] Error computing prompt vs intervention: {e}")

        if original_image is not None and intervention_image is not None:
            try:
                orig_resized = original_image.resize((224, 224), Image.LANCZOS)
                interv_resized = intervention_image.resize((224, 224), Image.LANCZOS)

                orig_tensor = pil_to_tensor(orig_resized).unsqueeze(0).to(device)
                interv_tensor = pil_to_tensor(interv_resized).unsqueeze(0).to(device)

                clip_model = metric.model

                with torch.no_grad():
                    orig_features = clip_model.get_image_features(orig_tensor)
                    interv_features = clip_model.get_image_features(interv_tensor)

                    orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
                    interv_features = interv_features / interv_features.norm(dim=-1, keepdim=True)

                    similarity = (orig_features @ interv_features.T).item()
                    results["image_similarity"] = (similarity + 1) * 50

            except Exception as e:
                print(f"[CLIP] Error computing image similarity: {e}")

    except Exception as e:
        print(f"[CLIP] Error initializing CLIP model: {e}")

    return results


#     """
#     Get emoji indicator and CSS class based on CLIP score.
#
#     For text-to-image: Higher scores (closer to 100) indicate better alignment
#     between the prompt and the generated image.
#
#     Args:
#         score: CLIP score (0-100) or None
#
#     Returns:
#         Tuple of (emoji, css_class)
#     """


def format_clip_scores(scores: dict) -> str:
    """
    Format CLIP scores as styled HTML for display in the dashboard.

    Args:
        scores: Dictionary with clip score results

    Returns:
        HTML formatted CLIP scores display
    """
    prompt_orig = scores.get("prompt_original")
    prompt_interv = scores.get("prompt_intervention")
    image_sim = scores.get("image_similarity")

    def format_score(score: float | None) -> str:
        if score is None:
            return '<span class="score-value score-na">—</span>'
        return f'<span class="score-value">{score:.1f}</span>'

    html = f"""
<div class="analysis-container clip-container">
    <div class="analysis-header">
        <p class="analysis-description">
            <strong>CLIP Score</strong> measures semantic similarity between text and images (0-100).
            Higher scores indicate better alignment. For text-to-image, scores of
            <strong>25-35</strong> typically indicate good prompt adherence.
        </p>
    </div>

    <div class="analysis-content">
        <div class="clip-scores-grid">
            <div class="clip-score-card">
                <div class="clip-score-label">🖼️ Original ↔ Intervention</div>
                <div class="clip-score-sublabel">Image-to-image similarity</div>
                {format_score(image_sim)}
            </div>

            <div class="clip-score-card">
                <div class="clip-score-label">📝 Prompt → Original</div>
                <div class="clip-score-sublabel">Text-to-image alignment</div>
                {format_score(prompt_orig)}
            </div>

            <div class="clip-score-card">
                <div class="clip-score-label">📝 Prompt → Intervention</div>
                <div class="clip-score-sublabel">Text-to-image alignment</div>
                {format_score(prompt_interv)}
            </div>
        </div>
    </div>
</div>
"""
    return html
