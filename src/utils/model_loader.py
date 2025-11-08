"""
Utility module for downloading and loading Stable Diffusion models.
Handles both local and remote (Google Drive) model loading.
"""

import os
from pathlib import Path
from typing import Optional
import gdown
import torch
from diffusers import StableDiffusionPipeline

class ModelLoader:
    DEFAULT_MODEL_ID = "sd-legacy/stable-diffusion-v1-5"

    def __init__(
        self, 
        gdrive_folder_url: str,
        model_name: str,
        project_root: Optional[Path] = None
    ):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = project_root
        
        self.gdrive_folder_url = gdrive_folder_url
        self.model_name = model_name
        
        # Set up cache path
        self.cache_dir = self.project_root / ".." / ".cache" / "models"
        self.model_path = self.cache_dir / self.model_name

    def download_model_from_gdrive(self) -> Path:
        """Downloads the model from Google Drive if not present locally."""
        if self.model_path.exists():
            print(f"Using existing model from: {self.model_path}")
            return self.model_path
            
        print("Downloading model from Google Drive...")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download directly to cache location
            gdown.download_folder(
                url=self.gdrive_folder_url, 
                output=str(self.model_path),
                quiet=False
            )
            
            print(f"Model downloaded to: {self.model_path}")
            return self.model_path

        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    def load_model(
        self, 
        device: str = "cuda",
        use_default_if_download_fails: bool = True
    ) -> StableDiffusionPipeline:
        """
        Loads the Stable Diffusion model, attempting to use the fine-tuned version first.
        
        Args:
            device: Device to load the model on ('cuda' or 'cpu')
            use_default_if_download_fails: If True, falls back to default SD 1.5 on failure
            
        Returns:
            StableDiffusionPipeline: The loaded model pipeline
        """
        try:
            model_path = self.download_model_from_gdrive()
            model_id = str(model_path)
        except Exception as e:
            if not use_default_if_download_fails:
                raise
            print(f"Failed to load fine-tuned model, falling back to {self.DEFAULT_MODEL_ID}: {e}")
            model_id = self.DEFAULT_MODEL_ID

        dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
        
        print(f"\nLoading model from: {model_id}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            dtype=dtype,
            safety_checker=None,
        ).to(device)
        
        return pipe