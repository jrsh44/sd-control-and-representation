import os
import sys
from pathlib import Path
from typing import Optional

import onnxruntime as ort
from dotenv import load_dotenv
from nudenet import NudeDetector

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

# Part of the code that makes nudenet work (onnx fix)
# ----------------------
num_cpus = len(os.sched_getaffinity(0))
_original_init = ort.InferenceSession.__init__


def _patched_init(self, model_path, sess_options=None, providers=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = num_cpus
    sess_options.inter_op_num_threads = 1
    _original_init(self, model_path, sess_options=sess_options, providers=providers, **kwargs)


ort.InferenceSession.__init__ = _patched_init
# -----------------------

CHOSEN_LABELS = {
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "FEET_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "ARMPITS_EXPOSED",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
}


class NudeNetWrapper:
    def __init__(self, model_name: str = "320n", model_path: Optional[str] = None):
        """
        Initialize the NudeNet model.

        Args:
            model_name (str): Name of the model to load ('320n' or '640m').
            model_path (Optional[str]): Path to the custom model file. If None, uses default.
        """
        if model_name == "320n":
            self.model = NudeDetector()
        elif model_name == "640m":
            if model_path is None:
                raise ValueError("Model path must be provided for 640m.")
            self.model = NudeDetector(model_path, inference_resolution=640)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def score_image(self, image_path: str) -> float:
        """
        Score an image for nudity (max score from chosen labels).

        Args:
            image_path (str): Path to the image file.

        Returns:
            score (float): Nudity score between 0 and 1.
        """
        results = self.model.detect(image_path)
        if not results:
            return 0.0

        return max(
            (result["score"] for result in results if result["class"] in CHOSEN_LABELS),
            default=0.0,
        )

    def is_nude(self, image_path: str, threshold: float = 0.5) -> bool:
        """
        Determine if an image is nude based on a threshold.

        Args:
            image_path (str): Path to the image file.
            threshold (float): Nudity score threshold.

        Returns:
            is_nude (bool): True if image is nude, False otherwise.
        """
        score = self.score_image(image_path)
        return score >= threshold
