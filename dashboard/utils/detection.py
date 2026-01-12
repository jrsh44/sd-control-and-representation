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


def detect_content(
    image: Image.Image,
    state: "DashboardState",
) -> tuple[dict, Image.Image | None]:
    """
    Run NudeNet detection on an image and create a censored version if needed.

    Args:
        image: PIL Image to analyze
        state: Dashboard state with nudenet_detector and temp_dir

    Returns:
        Tuple of (detection results dict, censored image or None)
    """
    if state.nudenet_detector is None:
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
        detections = state.nudenet_detector.detect(str(temp_path))

        # Analyze results
        has_unsafe = any(det["class"] in UNSAFE_LABELS and det["score"] > 0.5 for det in detections)

        # Create censored image if unsafe content detected
        censored_image = None
        if has_unsafe:
            try:
                # Use NudeNet's censor method to blur detected regions
                censored_path = state.temp_dir / f"censored_{int(time.time() * 1000)}.png"
                state.nudenet_detector.censor(
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
    Format NudeNet detection results as a comparison table in Markdown.

    Args:
        detection_orig: Detection results for original image
        detection_interv: Detection results for intervention image

    Returns:
        Markdown formatted comparison table
    """
    # Build comparison table
    lines = [
        '<div class="nudenet-scores-comparison">',
        "",
        "| Body Part | Original | Intervention |",
        "|-----------|----------|--------------|",
    ]

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

    if not all_labels:
        lines.append("| No detections | - | - |")
    else:
        # Sort by original score descending
        sorted_labels = sorted(all_labels, key=lambda x: orig_scores.get(x, 0.0), reverse=True)

        for label in sorted_labels:
            orig_val = orig_scores.get(label, 0.0)
            interv_val = interv_scores.get(label, 0.0)

            # Format with color coding
            orig_str = get_score_cell_text(orig_val)
            interv_str = get_score_cell_text(interv_val)

            # Add indicators for unsafe labels
            if label in UNSAFE_LABELS:
                label_display = f"⚠️ {label}"
            else:
                label_display = label

            lines.append(f"| {label_display} | {orig_str} | {interv_str} |")

    lines.append("")
    lines.append("</div>")

    return "\n".join(lines)


def get_score_cell_text(score: float) -> str:
    """Get formatted score text for table cell.

    Args:
        score: Detection score (0.0 to 1.0)

    Returns:
        Formatted score string
    """
    emoji = ""

    if score >= 0.75:
        emoji = "🔴"
    elif score >= 0.5:
        emoji = "🟠"
    elif score >= 0.25:
        emoji = "🟡"
    elif score >= 0.0:
        emoji = "🟢"

    return f"{emoji} {score:.2%}"
