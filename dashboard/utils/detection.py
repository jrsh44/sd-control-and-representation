"""
NudeNet Detection Utilities Module

Provides content detection and censoring functionality using NudeNet.
"""

import time
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    import sys
    from pathlib import Path as _Path

    _dashboard_dir = _Path(__file__).parent.parent
    if str(_dashboard_dir) not in sys.path:
        sys.path.insert(0, str(_dashboard_dir))
    from core.state import DashboardState


# All NudeNet detection classes
NUDENET_CLASSES = [
    "BUTTOCKS_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "ARMPITS_EXPOSED",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
]

# Unsafe labels that trigger censoring
UNSAFE_LABELS = [
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
]


def detect_nudity_coordinates(
    image: Image.Image,
    state: "DashboardState",
    detector=None,
) -> list[dict]:
    """
    Run NudeNet detection to get coordinates of nudity regions without censoring.

    Args:
        image: PIL Image to analyze
        state: Dashboard state with nudenet_detector and temp_dir
        detector: Optional NudeNet detector (uses state.nudenet_detector if None)

    Returns:
        List of detection dictionaries with 'class', 'score', and 'box' keys
        Box format: [x, y, width, height]
    """
    nudenet_detector = detector if detector is not None else state.nudenet_detector

    if nudenet_detector is None:
        return []

    try:
        # Save image temporarily for detection
        temp_path = state.temp_dir / f"detect_{int(time.time() * 1000)}.png"
        image.save(temp_path)

        # Run detection to get coordinates
        detections = nudenet_detector.detect(str(temp_path))

        # Clean up temp file
        temp_path.unlink(missing_ok=True)

        # Filter for unsafe labels with sufficient confidence
        unsafe_detections = [
            det for det in detections if det["class"] in UNSAFE_LABELS and det["score"] > 0.5
        ]

        return unsafe_detections

    except Exception as e:
        state.log(f"Detection error: {str(e)}", "error")
        return []


def apply_censorship_boxes(
    image: Image.Image,
    detections: list[dict],
    blur_intensity: int = 50,
) -> Image.Image:
    """
    Apply censorship to an image using pre-detected coordinates.
    Scales coordinates if image size differs from detection image.

    Args:
        image: PIL Image to censor
        detections: List of detection dicts with 'box' key [x, y, w, h]
        blur_intensity: Deprecated, kept for API compatibility

    Returns:
        PIL Image with censored regions (black boxes)
    """
    if not detections:
        return image

    import numpy as np

    # Convert PIL to numpy array
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]

    # Apply censorship to each detection
    for det in detections:
        box = det["box"]
        x, y, w, h = box

        # Ensure coordinates are within image bounds
        x = max(0, min(x, img_w))
        y = max(0, min(y, img_h))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w > 0 and h > 0:
            # Draw black box over the region
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            img_array[y1:y2, x1:x2] = 0  # Black color

    # Convert back to PIL
    return Image.fromarray(img_array)


def detect_content(
    image: Image.Image,
    state: "DashboardState",
    detector=None,
) -> tuple[dict, Image.Image | None]:
    """
    Run NudeNet detection on an image and create a censored version if needed.

    Args:
        image: PIL Image to analyze
        state: Dashboard state with nudenet_detector and temp_dir
        detector: Optional NudeNet detector (uses state.nudenet_detector if None)

    Returns:
        Tuple of (detection results dict, censored image or None)
    """
    # Use provided detector or fall back to state's detector
    nudenet_detector = detector if detector is not None else state.nudenet_detector

    if nudenet_detector is None:
        return {
            "detections": [],
            "has_unsafe": False,
            "summary": "NudeNet detector not loaded",
            "error": "Detector not initialized",
        }, None

    try:
        # Save image temporarily for detection
        temp_path = state.temp_dir / f"detect_{int(time.time() * 1000)}.png"
        image.save(temp_path)

        # Run detection
        detections = nudenet_detector.detect(str(temp_path))

        # Analyze results
        has_unsafe = any(det["class"] in UNSAFE_LABELS and det["score"] > 0.5 for det in detections)

        # Create censored image if unsafe content detected
        censored_image = None
        if has_unsafe:
            try:
                # Use NudeNet's censor method to blur detected regions
                censored_path = state.temp_dir / f"censored_{int(time.time() * 1000)}.png"
                nudenet_detector.censor(
                    str(temp_path),
                    classes=UNSAFE_LABELS,
                    output_path=str(censored_path),
                )
                censored_image = Image.open(censored_path)
                # Make a copy so we can delete the temp file
                censored_image = censored_image.copy()
                censored_path.unlink(missing_ok=True)
                state.log("Censored image created with blurred regions", "info")
            except Exception as e:
                state.log(f"Failed to create censored image: {str(e)}", "warning")

        # Clean up temp file
        temp_path.unlink(missing_ok=True)

        # Create summary
        if not detections:
            summary = "No sensitive content detected"
        elif has_unsafe:
            unsafe_count = sum(
                1 for det in detections if det["class"] in UNSAFE_LABELS and det["score"] > 0.5
            )
            summary = f"⚠️ {unsafe_count} unsafe region(s) detected and blurred"
        else:
            summary = f"✓ Safe ({len(detections)} region(s) analyzed)"

        return {
            "detections": detections,
            "has_unsafe": has_unsafe,
            "summary": summary,
            "total_detections": len(detections),
        }, censored_image

    except Exception as e:
        state.log(f"Detection error: {str(e)}", "error")
        return {
            "detections": [],
            "has_unsafe": False,
            "summary": f"Detection failed: {str(e)}",
            "error": str(e),
        }, None


def format_nudenet_comparison(
    detection_orig: dict | None,
    detection_interv: dict | None,
) -> str:
    """
    Format NudeNet detection results as a comparison table in HTML.

    Args:
        detection_orig: Detection results for original image
        detection_interv: Detection results for intervention image

    Returns:
        HTML formatted comparison table
    """
    # Get all detections from both
    orig_scores = {}
    interv_scores = {}

    if detection_orig and "detections" in detection_orig:
        for det in detection_orig["detections"]:
            label = det.get("class", "Unknown")
            score = det.get("score", 0.0)
            if label not in orig_scores or score > orig_scores[label]:
                orig_scores[label] = score

    if detection_interv and "detections" in detection_interv:
        for det in detection_interv["detections"]:
            label = det.get("class", "Unknown")
            score = det.get("score", 0.0)
            if label not in interv_scores or score > interv_scores[label]:
                interv_scores[label] = score

    # Combine all labels
    all_labels = set(orig_scores.keys()) | set(interv_scores.keys())

    # Build table rows
    if not all_labels:
        table_rows = '<tr><td colspan="3" style="text-align: center; color: var(--text-secondary);">No detections</td></tr>'
    else:
        # Sort by original score descending
        sorted_labels = sorted(all_labels, key=lambda x: orig_scores.get(x, 0.0), reverse=True)
        rows = []
        for label in sorted_labels:
            orig_val = orig_scores.get(label, 0.0)
            interv_val = interv_scores.get(label, 0.0)

            # Format with color coding
            orig_str = get_score_cell_html(orig_val)
            interv_str = get_score_cell_html(interv_val)

            # Add indicators for unsafe labels
            if label in UNSAFE_LABELS:
                label_display = f'<span class="unsafe-label">⚠️ {label}</span>'
            else:
                label_display = label

            rows.append(
                f"<tr><td>{label_display}</td><td>{orig_str}</td><td>{interv_str}</td></tr>"
            )
        table_rows = "\n                ".join(rows)

    html = f"""
<div class="analysis-container nudenet-container">
    <div class="analysis-header">
        <p class="analysis-description">
            <strong>NudeNet</strong> is an NSFW object detection model that identifies exposed body parts in images.
            Each detection includes a confidence score (0-100%) indicating detection certainty.
            Scores above <strong>50%</strong> are considered significant. ⚠️ marks unsafe content categories.
        </p>
    </div>

    <div class="analysis-content">
        <table class="detection-table">
            <thead>
                <tr>
                    <th>Body Part</th>
                    <th>Original</th>
                    <th>Intervention</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
</div>
"""
    return html


def get_score_cell_html(score: float) -> str:
    """Get formatted score HTML for table cell.

    Args:
        score: Detection score (0.0 to 1.0)

    Returns:
        HTML formatted score string with color indicator
    """
    if score >= 0.75:
        css_class = "score-high"
        emoji = "🔴"
    elif score >= 0.5:
        css_class = "score-medium"
        emoji = "🟠"
    elif score >= 0.25:
        css_class = "score-low"
        emoji = "🟡"
    else:
        css_class = "score-safe"
        emoji = "🟢"

    return f'<span class="score-value {css_class}">{emoji} {score:.0%}</span>'


# Keep old function for backward compatibility
def get_score_cell_text(score: float) -> str:
    """Get formatted score text for table cell (deprecated, use get_score_cell_html).

    Args:
        score: Detection score (0.0 to 1.0)

    Returns:
        Formatted score string
    """
    return get_score_cell_html(score)
