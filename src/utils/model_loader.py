"""
Utility module for downloading and loading Stable Diffusion models.
Handles both local and remote (Google Drive and Hugging Face) model loading.
"""

import os
import sys
from pathlib import Path
from typing import Literal, Optional, Union

import gdown
import torch
from dotenv import load_dotenv

from src.utils.model_enum import ModelEnum

# Load environment variables
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline


class ModelLoader:
    DEFAULT_MODEL_ID = "sd-legacy/stable-diffusion-v1-5"

    def __init__(
        self,
        model_enum: Optional["ModelEnum"] = None,
        model_name: Optional[str] = None,
        model_source: Optional[Literal["gdrive", "huggingface"]] = None,
        model_id: Optional[str] = None,
        project_root: Optional[Path] = None,
        pipeline_type: Optional[Literal["sd", "sd3"]] = "sd",
    ):
        """
        Initialize ModelLoader to load a model from local cache, Google Drive, or Hugging Face.

        Args:
            model_name: Name to use for local caching and lookup
            model_source: Optional - Source of the model ("gdrive" or "huggingface")
            model_id: Optional - For gdrive - the folder URL, for huggingface - the model ID
            project_root: Optional project root path

        If model_id is not provided, will try to find existing model in cache with model_name
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = project_root

        if model_enum is not None:
            self.model_source = model_enum.source
            self.model_id = model_enum.model_id
            self.model_name = model_enum.name
            self.pipeline_type = model_enum.pipeline_type
        else:
            self.model_source = model_source
            self.model_id = model_id
            self.model_name = model_name
            self.pipeline_type = pipeline_type

        try:
            self.cache_dir = Path(os.environ["CACHE_DIR"]) / "models"
        except KeyError:
            print("Specify CACHE_DIR in your .env file or environment variables.")
            sys.exit(1)
        self.model_path = self.cache_dir / self.model_name

    def _download_from_gdrive(self) -> Path:
        """Downloads the model from Google Drive if not present locally."""
        if self.model_path.exists():
            print(f"Using existing model from: {self.model_path}")
            return self.model_path

        print("Downloading model from Google Drive...")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            gdown.download_folder(url=self.model_id, output=str(self.model_path), quiet=False)

            print(f"Model downloaded to: {self.model_path}")
            return self.model_path

        except Exception as e:
            print(f"Error downloading model from Google Drive: {e}")
            raise

    def get_model_path(self) -> str:
        """Gets the appropriate model path/ID based on the source."""
        if self.model_path.exists():
            print(f"Using existing model from cache: {self.model_path}")
            return str(self.model_path)

        if not self.model_source or not self.model_id:
            raise ValueError(
                f"No model found in cache for {self.model_name} and no source specified"
            )

        if self.model_source == "gdrive":
            return str(self._download_from_gdrive())
        else:
            return self.model_id

    def load_model(
        self, device: str = "cuda", use_default_if_download_fails: bool = True
    ) -> Union[StableDiffusionPipeline, StableDiffusion3Pipeline]:
        """
        Loads the Stable Diffusion model from the specified source.

        Args:
            device: Device to load the model on ('cuda' or 'cpu')
            use_default_if_download_fails: If True, falls back to default SD 1.5 on failure

        Returns:
            Either StableDiffusionPipeline or StableDiffusion3Pipeline depending on the model type
        """
        try:
            model_id = self.get_model_path()
        except Exception as e:
            if not use_default_if_download_fails:
                raise
            print(f"Failed to load model, falling back to {self.DEFAULT_MODEL_ID}: {e}")
            model_id = self.DEFAULT_MODEL_ID

        dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32

        print(f"\nLoading model from: {model_id}")

        if self.pipeline_type == "sd3":
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
            ).to(device)
        elif self.pipeline_type == "sd":
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
            ).to(device)
        else:
            raise ValueError(f"Unsupported pipeline type: {self.pipeline_type}")

        return pipe
