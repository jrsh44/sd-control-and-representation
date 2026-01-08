"""Unit tests for src/utils/model_loader.py"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.config import ModelRegistry
from src.utils.model_loader import ModelLoader


def test_model_loader_initialization():
    """Test ModelLoader initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["CACHE_DIR"] = tmpdir

        loader = ModelLoader(ModelRegistry.SD_V1_5)

        assert loader.model_source == "huggingface"
        assert loader.model_id == "sd-legacy/stable-diffusion-v1-5"
        assert loader.model_name == "SD_V1_5"
        assert loader.pipeline_type == "sd_v1_5"
        assert loader.cache_dir == Path(tmpdir) / "models"


def test_model_loader_initialization_custom_root():
    """Test ModelLoader with custom project root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["CACHE_DIR"] = tmpdir
        custom_root = Path(tmpdir) / "custom"
        custom_root.mkdir()

        loader = ModelLoader(ModelRegistry.SD_V1_5, project_root=custom_root)

        assert loader.project_root == custom_root


def test_model_loader_no_cache_dir():
    """Test ModelLoader fails without CACHE_DIR."""
    # Save original and remove CACHE_DIR
    original = os.environ.get("CACHE_DIR")
    if "CACHE_DIR" in os.environ:
        del os.environ["CACHE_DIR"]

    try:
        with pytest.raises(SystemExit):
            ModelLoader(ModelRegistry.SD_V1_5)
    finally:
        # Restore original
        if original:
            os.environ["CACHE_DIR"] = original


def test_get_model_path_huggingface():
    """Test get_model_path for Hugging Face model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["CACHE_DIR"] = tmpdir

        loader = ModelLoader(ModelRegistry.SD_V1_5)
        model_path = loader.get_model_path()

        # For HF models, should return model_id directly if not cached
        assert model_path == "sd-legacy/stable-diffusion-v1-5"


def test_get_model_path_cached():
    """Test get_model_path returns local path when cached."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["CACHE_DIR"] = tmpdir

        loader = ModelLoader(ModelRegistry.SD_V1_5)
        # Create cache directory
        loader.model_path.mkdir(parents=True)

        model_path = loader.get_model_path()

        # Should return local path when cache exists
        assert model_path == str(loader.model_path)


@patch("src.utils.model_loader.gdown.download_folder")
def test_download_from_gdrive_not_exists(mock_gdown):
    """Test downloading from Google Drive when model doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["CACHE_DIR"] = tmpdir

        loader = ModelLoader(ModelRegistry.FINETUNED_SAEURON)

        # Mock successful download
        mock_gdown.return_value = None

        result = loader._download_from_gdrive()

        # Should create directory and call gdown
        assert result == loader.model_path
        mock_gdown.assert_called_once()


@patch("src.utils.model_loader.gdown.download_folder")
def test_download_from_gdrive_already_exists(mock_gdown):
    """Test downloading from Google Drive when model already exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["CACHE_DIR"] = tmpdir

        loader = ModelLoader(ModelRegistry.FINETUNED_SAEURON)
        loader.model_path.mkdir(parents=True)

        result = loader._download_from_gdrive()

        # Should not call gdown if already exists
        assert result == loader.model_path
        mock_gdown.assert_not_called()


@patch("src.utils.model_loader.gdown.download_folder")
def test_download_from_gdrive_error(mock_gdown):
    """Test downloading from Google Drive handles errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["CACHE_DIR"] = tmpdir

        loader = ModelLoader(ModelRegistry.FINETUNED_SAEURON)

        # Mock download error
        mock_gdown.side_effect = Exception("Download failed")

        with pytest.raises(Exception, match="Download failed"):
            loader._download_from_gdrive()


def test_model_loader_different_models():
    """Test ModelLoader with different model types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["CACHE_DIR"] = tmpdir

        # Test SD v1.5
        loader_v1_5 = ModelLoader(ModelRegistry.SD_V1_5)
        assert loader_v1_5.pipeline_type == "sd_v1_5"

        # Test SD v3
        loader_v3 = ModelLoader(ModelRegistry.SD_V3)
        assert loader_v3.pipeline_type == "sd_v3"

        # Test finetuned
        loader_ft = ModelLoader(ModelRegistry.FINETUNED_SAEURON)
        assert loader_ft.model_source == "gdrive"
