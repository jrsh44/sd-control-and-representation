"""
Utility module for downloading and loading Stable Diffusion models.
Handles both local and remote (Google Drive and Hugging Face) model loading.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union

import gdown
import torch
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline  # noqa: E402

from src.models.config import ModelRegistry  # noqa: E402


class ModelLoader:
    def __init__(
        self,
        model_enum: ModelRegistry,
        project_root: Optional[Path] = None,
    ):
        """
        Initialize ModelLoader to load a model from local cache, Google Drive, or Hugging Face.

        Args:
            model_enum: ModelRegistry enum specifying the model to load
            project_root: Optional project root path
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = project_root

        self.model_source = model_enum.source
        self.model_id = model_enum.model_id
        self.model_name = model_enum.name
        self.pipeline_type = model_enum.pipeline_type

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

        if self.model_source == "gdrive":
            return str(self._download_from_gdrive())
        else:
            return self.model_id

    def load_model(
        self, device: str = "cuda"
    ) -> Union[StableDiffusionPipeline, StableDiffusion3Pipeline]:
        """
        Loads the Stable Diffusion model from the specified source.

        Args:
            device: Device to load the model on ('cuda' or 'cpu')

        Returns:
            Union[StableDiffusionPipeline, StableDiffusion3Pipeline]: The loaded model pipeline.
            Returns StableDiffusionPipeline for SD1.5 models and
            StableDiffusion3Pipeline for SD3 models.
        """
        try:
            model_id = self.get_model_path()
        except Exception as e:
            raise FileNotFoundError(f"Failed to locate or download model: {str(e)}") from e

        dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32

        print(f"\nLoading model from: {model_id}")

        if self.pipeline_type == "sd_v1_5":
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
            ).to(device)
        elif self.pipeline_type == "sd_v3":
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
            ).to(device)
        else:
            raise ValueError(f"Unsupported pipeline type: {self.pipeline_type}")

        return pipe
