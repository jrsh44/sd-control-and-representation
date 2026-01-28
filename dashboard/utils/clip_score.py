"""
CLIP Score Utilities Module

Provides CLIP-based similarity metrics for evaluating generated images.
Uses torchmetrics CLIPScore to calculate:
- Image-to-image similarity (Original vs Intervention)
- Text-to-image similarity (Prompt vs Original, Prompt vs Intervention)

CLIP Score is a reference-free metric that correlates highly with human judgment.
Scores range from 0-100, with higher scores indicating better alignment.
"""

from typing import TYPE_CHECKING

import torch
from PIL import Image
from torchmetrics.multimodal import CLIPScore
from torchvision import transforms

if TYPE_CHECKING:
    from core.state import DashboardState


# Module-level fallback cache (used when state is not available)
_clip_model = None
_clip_device = None


def get_clip_model(device: str = "cpu", state: "DashboardState | None" = None) -> CLIPScore:
    """
    Get or initialize the CLIP model.

    Uses the state's clip_model if available, otherwise falls back to module cache.

    Args:
        device: Device to load model on ('cpu' or 'cuda')
        state: Optional dashboard state that may have pre-loaded CLIP model

    Returns:
        CLIPScore metric instance
    """
    global _clip_model, _clip_device

    # Try to use state's pre-loaded model first
    if state is not None and state.clip_model is not None:
        # Move to correct device if needed
        if hasattr(state, "clip_device") and state.clip_device != device:
            state.clip_model = state.clip_model.to(device)
            state.clip_device = device
        return state.clip_model

    # Fall back to module-level cache
    if _clip_model is None or _clip_device != device:
        _clip_model = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
        _clip_model = _clip_model.to(device)
        _clip_device = device

    return _clip_model


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to tensor format expected by CLIPScore.

    Args:
        image: PIL Image to convert

    Returns:
        Tensor of shape [C, H, W] with values in [0, 255]
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to tensor [C, H, W] in range [0, 1]
    transform = transforms.ToTensor()
    tensor = transform(image)

    # Scale to [0, 255] as expected by CLIPScore
    tensor = (tensor * 255).to(torch.uint8)

    return tensor


def calculate_clip_scores(
    prompt: str,
    original_image: Image.Image | None,
    intervention_image: Image.Image | None,
    device: str = "cpu",
    state: "DashboardState | None" = None,
) -> dict:
    """
    Calculate CLIP scores for prompt-image and image-image comparisons.

    Args:
        prompt: Text prompt used for generation
        original_image: Original generated image (PIL)
        intervention_image: Image generated with intervention (PIL)
        device: Device for computation ('cpu' or 'cuda')
        state: Optional dashboard state with pre-loaded CLIP model

    Returns:
        Dictionary with scores:
        - prompt_original: CLIP score between prompt and original image
        - prompt_intervention: CLIP score between prompt and intervention image
        - image_similarity: CLIP score between original and intervention images
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

        # Calculate prompt vs original image
        if original_image is not None:
            try:
                orig_tensor = pil_to_tensor(original_image)
                metric.reset()
                metric.update(orig_tensor, prompt)
                score = metric.compute()
                results["prompt_original"] = float(score.item())
            except Exception as e:
                print(f"[CLIP] Error computing prompt vs original: {e}")

        # Calculate prompt vs intervention image
        if intervention_image is not None:
            try:
                interv_tensor = pil_to_tensor(intervention_image)
                metric.reset()
                metric.update(interv_tensor, prompt)
                score = metric.compute()
                results["prompt_intervention"] = float(score.item())
            except Exception as e:
                print(f"[CLIP] Error computing prompt vs intervention: {e}")

        # Calculate image-to-image similarity using embeddings
        if original_image is not None and intervention_image is not None:
            try:
                # Get CLIP embeddings for both images
                orig_tensor = pil_to_tensor(original_image).unsqueeze(0).to(device)
                interv_tensor = pil_to_tensor(intervention_image).unsqueeze(0).to(device)

                # Access the underlying CLIP model to get image embeddings
                clip_model = metric.model

                # Get image features
                with torch.no_grad():
                    orig_features = clip_model.get_image_features(orig_tensor)
                    interv_features = clip_model.get_image_features(interv_tensor)

                    # Normalize features
                    orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
                    interv_features = interv_features / interv_features.norm(dim=-1, keepdim=True)

                    # Compute cosine similarity and scale to 0-100
                    similarity = (orig_features @ interv_features.T).item()
                    # Convert from [-1, 1] to [0, 100]
                    results["image_similarity"] = (similarity + 1) * 50

            except Exception as e:
                print(f"[CLIP] Error computing image similarity: {e}")

    except Exception as e:
        print(f"[CLIP] Error initializing CLIP model: {e}")

    return results


# def get_score_indicator(score: float | None) -> tuple[str, str]:
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
#     if score is None:
#         return "—", "score-na"
#
#     # CLIP scores typically range:
#     # - Text-to-image: 20-35 is typical for good alignment
#     # - Image-to-image: Can be higher (60-100) for similar images
#     if score >= 30:
#         return "🟢", "score-excellent"
#     elif score >= 25:
#         return "🟡", "score-good"
#     elif score >= 20:
#         return "🟠", "score-moderate"
#     else:
#         return "🔴", "score-low"


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

    # Format individual scores
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
