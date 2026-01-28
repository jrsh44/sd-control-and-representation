"""
Dashboard Tests - Comprehensive Testing Suite
Test suite for the SD Control & Representation dashboard

Tests the dashboard components:
- core/state.py: State management classes
- core/model_loader.py: Model loading functions (mocked)
- config/sae_config_loader.py: SAE configuration loading
- utils/: Utility modules (cuda, clip_score, detection, heatmap)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from PIL import Image

# Add dashboard to path for imports
dashboard_dir = Path(__file__).parent.parent / "dashboard"
if str(dashboard_dir) not in sys.path:
    sys.path.insert(0, str(dashboard_dir))


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Test DashboardState class from state.py"""

    def test_state_initialization(self):
        """Test state initializes correctly"""
        from core.state import DashboardState, ModelLoadState, SystemState

        state = DashboardState()
        assert state.system_state == SystemState.IDLE
        assert state.sd_pipe is None
        assert state.sae_model is None
        assert state.nudenet_detector is None
        assert len(state.model_states) == 3
        assert state.model_states["sd_base"] == ModelLoadState.NOT_LOADED

    def test_log_message(self):
        """Test logging functionality"""
        from core.state import DashboardState

        state = DashboardState()
        result = state.log("Test message", "info")
        assert "Test message" in result
        assert "[INFO]" in result
        assert len(state.logs) > 0

    def test_model_state_update(self):
        """Test model state updates"""
        from core.state import DashboardState, ModelLoadState

        state = DashboardState()
        state.set_model_state("sd_base", ModelLoadState.LOADED, 5.2)
        assert state.model_states["sd_base"] == ModelLoadState.LOADED
        assert state.load_times["sd_base"] == 5.2

    def test_get_model_status_text(self):
        """Test model status text generation"""
        from core.state import DashboardState

        state = DashboardState()
        status_text = state.get_model_status_text()
        assert isinstance(status_text, str)
        assert "MODEL STATUS" in status_text

    def test_generation_progress(self):
        """Test generation progress tracking"""
        import time

        from core.state import GenerationProgress

        progress = GenerationProgress(
            phase="original",
            current_step=25,
            total_steps=50,
            start_time=time.time() - 10,
        )
        assert progress.progress == 0.5
        assert progress.elapsed_time >= 10
        assert progress.format_time(65) == "01:05"

    def test_system_state_transitions(self):
        """Test valid system state transitions"""
        from core.state import DashboardState, SystemState

        state = DashboardState()
        state.system_state = SystemState.LOADING_MODEL
        assert state.system_state == SystemState.LOADING_MODEL

        state.system_state = SystemState.GENERATING
        assert state.system_state == SystemState.GENERATING

        state.system_state = SystemState.IDLE
        assert state.system_state == SystemState.IDLE

    def test_multiple_model_states(self):
        """Test tracking multiple model states"""
        from core.state import DashboardState, ModelLoadState

        state = DashboardState()
        state.set_model_state("sd_base", ModelLoadState.LOADED, 3.0)
        state.set_model_state("sae", ModelLoadState.LOADED, 2.5)
        state.set_model_state("nudenet", ModelLoadState.LOADED, 1.0)

        assert all(s == ModelLoadState.LOADED for s in state.model_states.values())
        assert state.load_times["sd_base"] == 3.0
        assert state.load_times["sae"] == 2.5
        assert state.load_times["nudenet"] == 1.0

    def test_temp_dir_creation(self):
        """Test temporary directory is created"""
        from core.state import DashboardState

        state = DashboardState()
        assert state.temp_dir is not None
        assert state.temp_dir.exists()
        assert state.temp_dir.is_dir()


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfigLoader:
    """Test SAE configuration loading from sae_config_loader.py"""

    def test_load_sae_config(self):
        """Test loading SAE configuration"""
        from config.sae_config_loader import load_sae_config

        config = load_sae_config()
        assert config is not None
        assert hasattr(config, "sae_models")
        assert isinstance(config.sae_models, list)

    def test_get_sae_model_choices(self):
        """Test getting SAE model choices"""
        from config.sae_config_loader import get_sae_model_choices, load_sae_config

        config = load_sae_config()
        choices = get_sae_model_choices(config)

        assert isinstance(choices, list)
        if len(choices) > 0:
            assert all(isinstance(choice, tuple) for choice in choices)
            assert all(len(choice) == 2 for choice in choices)

    def test_get_concept_choices(self):
        """Test getting concept choices for an SAE model"""
        from config.sae_config_loader import get_concept_choices, load_sae_config

        config = load_sae_config()
        if len(config.sae_models) > 0:
            sae_model_id = config.sae_models[0].id
            choices = get_concept_choices(config, sae_model_id)

            assert isinstance(choices, list)
            if len(choices) > 0:
                assert all(isinstance(choice, tuple) for choice in choices)
                assert all(len(choice) == 3 for choice in choices)  # (name, id, description)

    def test_get_layer_id(self):
        """Test getting layer ID for an SAE model"""
        from config.sae_config_loader import get_layer_id, load_sae_config

        config = load_sae_config()
        if len(config.sae_models) > 0:
            sae_model_id = config.sae_models[0].id
            layer_id = get_layer_id(config, sae_model_id)
            assert isinstance(layer_id, str)

    def test_get_feature_sums_path(self):
        """Test getting feature sums path"""
        from config.sae_config_loader import get_feature_sums_path, load_sae_config

        config = load_sae_config()
        if len(config.sae_models) > 0:
            sae_model_id = config.sae_models[0].id
            path = get_feature_sums_path(config, sae_model_id)
            assert path is None or isinstance(path, Path)

    def test_get_sae_hyperparameters(self):
        """Test getting SAE hyperparameters"""
        from config.sae_config_loader import get_sae_hyperparameters, load_sae_config

        config = load_sae_config()
        if len(config.sae_models) > 0:
            sae_model_id = config.sae_models[0].id
            hyperparams = get_sae_hyperparameters(config, sae_model_id)
            assert hyperparams is None or isinstance(hyperparams, dict)

    def test_get_model_path(self):
        """Test getting model path"""
        from config.sae_config_loader import get_model_path, load_sae_config

        config = load_sae_config()
        if len(config.sae_models) > 0:
            sae_model_id = config.sae_models[0].id
            path = get_model_path(config, sae_model_id)
            assert path is None or isinstance(path, Path)


# =============================================================================
# CUDA Utilities Tests
# =============================================================================


class TestCudaUtils:
    """Test CUDA utility functions"""

    def test_get_gpu_compute_capability(self):
        """Test GPU compute capability detection"""
        from utils.cuda import get_gpu_compute_capability

        result = get_gpu_compute_capability()
        # Result can be None (no GPU) or tuple of (major, minor)
        assert result is None or (isinstance(result, tuple) and len(result) == 2)

    def test_cuda_constants(self):
        """Test CUDA module constants are defined"""
        from utils.cuda import (
            CUDA_COMPATIBLE,
            CUDA_FLASH_ATTENTION_OK,
            CUDA_STATUS,
            GPU_COMPUTE_CAPABILITY,
        )

        assert isinstance(CUDA_COMPATIBLE, bool)
        assert isinstance(CUDA_FLASH_ATTENTION_OK, bool)
        assert isinstance(CUDA_STATUS, str)
        # GPU_COMPUTE_CAPABILITY can be None or tuple

    def test_get_pytorch_supported_cuda_archs(self):
        """Test getting PyTorch supported CUDA architectures"""
        from utils.cuda import get_pytorch_supported_cuda_archs

        archs = get_pytorch_supported_cuda_archs()
        assert isinstance(archs, list)
        assert all(isinstance(arch, int) for arch in archs)


# =============================================================================
# CLIP Score Tests
# =============================================================================


class TestClipScore:
    """Test CLIP score utilities"""

    def test_pil_to_tensor(self):
        """Test PIL image to tensor conversion"""
        from utils.clip_score import pil_to_tensor

        # Create a test image
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        tensor = pil_to_tensor(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)
        assert tensor.dtype == torch.uint8

    def test_pil_to_tensor_grayscale(self):
        """Test PIL grayscale image conversion"""
        from utils.clip_score import pil_to_tensor

        # Create a grayscale image
        img = Image.new("L", (64, 64), color=128)
        tensor = pil_to_tensor(img)

        # Should be converted to RGB
        assert tensor.shape == (3, 64, 64)

    @patch("utils.clip_score.CLIPScore")
    def test_get_clip_model_initialization(self, mock_clip_score):
        """Test CLIP model initialization"""
        from utils.clip_score import get_clip_model

        mock_model = MagicMock()
        mock_clip_score.return_value = mock_model

        model = get_clip_model(device="cpu")
        assert model is not None
        mock_clip_score.assert_called_once()

    @patch("utils.clip_score.get_clip_model")
    def test_calculate_clip_scores_both_images(self, mock_get_clip_model):
        """Test calculating CLIP scores with both images"""
        from utils.clip_score import calculate_clip_scores

        # Mock CLIP model
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([75.0])
        mock_get_clip_model.return_value = mock_model

        # Create test images
        img1 = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img2 = Image.new("RGB", (64, 64), color=(0, 255, 0))

        scores = calculate_clip_scores("a red image", img1, img2, device="cpu")

        assert isinstance(scores, dict)
        assert "prompt_original" in scores
        assert "prompt_intervention" in scores
        assert "image_similarity" in scores

    def test_format_clip_scores(self):
        """Test formatting CLIP scores"""
        from utils.clip_score import format_clip_scores

        scores = {
            "prompt_original": 75.5,
            "prompt_intervention": 65.3,
            "image_similarity": 85.7,
        }

        formatted = format_clip_scores(scores)
        assert isinstance(formatted, str)
        assert "75.5" in formatted
        assert "65.3" in formatted
        assert "85.7" in formatted


# =============================================================================
# Detection Utilities Tests
# =============================================================================


class TestDetectionUtils:
    """Test NudeNet detection utilities"""

    def test_nudenet_classes_defined(self):
        """Test that NudeNet classes are defined"""
        from utils.detection import NUDENET_CLASSES, UNSAFE_LABELS

        assert isinstance(NUDENET_CLASSES, list)
        assert isinstance(UNSAFE_LABELS, list)
        assert len(NUDENET_CLASSES) > 0
        assert len(UNSAFE_LABELS) > 0
        assert all(label in NUDENET_CLASSES for label in UNSAFE_LABELS)

    def test_detect_nudity_coordinates_no_detector(self):
        """Test detection with no detector returns empty list"""
        from core.state import DashboardState
        from utils.detection import detect_nudity_coordinates

        state = DashboardState()
        state.nudenet_detector = None

        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        detections = detect_nudity_coordinates(img, state)

        assert isinstance(detections, list)
        assert len(detections) == 0

    def test_apply_censorship_boxes_no_detections(self):
        """Test censorship with no detections returns original image"""
        from utils.detection import apply_censorship_boxes

        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        censored = apply_censorship_boxes(img, [])

        assert isinstance(censored, Image.Image)
        assert censored.size == img.size

    def test_format_nudenet_comparison_no_detections(self):
        """Test formatting comparison with no detections"""
        from utils.detection import format_nudenet_comparison

        result = format_nudenet_comparison([], [])
        assert isinstance(result, str)
        assert "Original" in result
        assert "Intervention" in result


# =============================================================================
# Heatmap Utilities Tests
# =============================================================================


class TestHeatmapUtils:
    """Test heatmap generation utilities"""

    @patch("utils.heatmap.torch")
    def test_decode_latent_to_image(self, mock_torch):
        """Test decoding latent to image"""
        from utils.heatmap import decode_latent_to_image

        # Mock pipeline with VAE
        mock_pipe = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.vae = mock_vae
        mock_vae.decode.return_value.sample = torch.randn(1, 3, 512, 512)

        latent = torch.randn(1, 4, 64, 64)
        image = decode_latent_to_image(latent, mock_pipe, "cpu")

        assert isinstance(image, Image.Image)

    def test_collect_activations_from_representations(self):
        """Test collecting activations from representations"""
        from utils.heatmap import collect_activations_from_representations

        # Mock SAE model
        mock_sae = MagicMock()
        mock_sae.return_value = (
            torch.randn(64, 1280),  # reconstructed
            torch.randn(64, 100),  # codes (activations)
            torch.randn(64, 100),  # indices
        )

        # Create mock representations
        representations = torch.randn(5, 1, 64, 1280)
        timesteps = [0, 1, 2]

        activations = collect_activations_from_representations(
            representations, mock_sae, timesteps, top_k_features=3, device="cpu"
        )

        assert isinstance(activations, dict)
        assert len(activations) == 3  # 3 timesteps


# =============================================================================
# Model Loader Tests (Mocked)
# =============================================================================


class TestModelLoader:
    """Test model loading functions (with mocking to avoid GPU requirements)"""

    @patch("diffusers.StableDiffusionPipeline")
    def test_load_sd_model_cpu(self, mock_pipeline_class):
        """Test loading SD model on CPU"""
        from core.model_loader import load_sd_model
        from core.state import DashboardState

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.device = "cpu"
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline

        state = DashboardState()
        pipe = load_sd_model(state, use_gpu=False)

        assert pipe is not None
        assert state.sd_pipe is not None
        mock_pipeline_class.from_pretrained.assert_called_once()

    @patch("torchmetrics.multimodal.CLIPScore")
    def test_load_clip_model(self, mock_clip_score_class):
        """Test loading CLIP model"""
        from core.model_loader import load_clip_model
        from core.state import DashboardState

        # Mock CLIP model
        mock_clip = MagicMock()
        mock_clip_score_class.return_value = mock_clip
        mock_clip.to.return_value = mock_clip

        state = DashboardState()
        clip_model = load_clip_model(state, device="cpu")

        assert clip_model is not None
        assert state.clip_model is not None
        mock_clip_score_class.assert_called_once()

    @patch("nudenet.NudeDetector")
    def test_load_nudenet_model(self, mock_nudenet_class):
        """Test loading NudeNet model"""
        from core.model_loader import load_nudenet_model
        from core.state import DashboardState

        # Mock NudeNet detector
        mock_detector = MagicMock()
        mock_nudenet_class.return_value = mock_detector

        state = DashboardState()
        detector = load_nudenet_model(state)

        assert detector is not None
        assert state.nudenet_detector is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full workflows"""

    def test_state_lifecycle(self):
        """Test complete state lifecycle"""
        from core.state import DashboardState, ModelLoadState, SystemState

        state = DashboardState()

        # Initial state
        assert state.system_state == SystemState.IDLE

        # Update system state
        state.system_state = SystemState.LOADING_MODEL
        assert state.system_state == SystemState.LOADING_MODEL

        # Update model state
        state.set_model_state("sd_base", ModelLoadState.LOADING)
        state.set_model_state("sd_base", ModelLoadState.LOADED, 3.5)

        # Check status
        status = state.get_model_status_text()
        assert "Loaded" in status or "●" in status

    def test_config_and_state_integration(self):
        """Test config loading and state management together"""
        from config.sae_config_loader import load_sae_config
        from core.state import DashboardState

        config = load_sae_config()
        state = DashboardState()

        assert config is not None
        assert state is not None
        assert state.system_state.value == "idle"

    def test_logging_workflow(self):
        """Test logging multiple messages"""
        from core.state import DashboardState

        state = DashboardState()

        # Log different types
        state.log("Info message", "info")
        state.log("Warning message", "warning")
        state.log("Error message", "error")
        state.log("Success message", "success")

        assert len(state.logs) >= 4
        log_text = "\n".join(state.logs)
        assert "INFO" in log_text
        assert "WARNING" in log_text
        assert "ERROR" in log_text
        assert "SUCCESS" in log_text


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
