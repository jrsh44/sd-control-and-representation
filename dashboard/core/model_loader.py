"""
Model Loading Module

Provides functions to load SD, SAE, and NudeNet models.
"""

# Use absolute import with dashboard prefix
import sys
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Any

import torch

_dashboard_dir = _Path(__file__).parent.parent
if str(_dashboard_dir) not in sys.path:
    sys.path.insert(0, str(_dashboard_dir))

from utils.cuda import CUDA_COMPATIBLE, CUDA_FLASH_ATTENTION_OK  # noqa: E402

if TYPE_CHECKING:
    from core.state import DashboardState


def load_sd_model(
    state: "DashboardState",
    use_gpu: bool = True,
    model_id: str = "sd-legacy/stable-diffusion-v1-5",
) -> Any:
    """Load Stable Diffusion v1.5 model.

    Args:
        state: Dashboard state to store the model
        use_gpu: Whether to use GPU if available
        model_id: HuggingFace model ID

    Returns:
        Loaded StableDiffusionPipeline
    """
    state.log(f"Loading SD from {model_id}...", "loading")

    try:
        from diffusers import StableDiffusionPipeline

        # Determine device based on user choice and compatibility
        if use_gpu and CUDA_COMPATIBLE:
            device = "cuda"
            dtype = torch.float16
            state.log("Using GPU with float16 precision", "info")
            if not CUDA_FLASH_ATTENTION_OK:
                state.log("Flash Attention disabled (GPU SM < 80)", "info")
        else:
            device = "cpu"
            dtype = torch.float32
            if use_gpu and not CUDA_COMPATIBLE:
                from utils.cuda import CUDA_STATUS

                state.log(f"GPU requested but not compatible: {CUDA_STATUS}", "warning")
            state.log("Using CPU with float32 precision", "info")

        state.log(f"Device: {device}, Dtype: {dtype}", "info")

        # Load pipeline
        state.log("Downloading/loading model weights...", "loading")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        state.log("Moving model to device...", "loading")
        pipe = pipe.to(device)

        # Enable optimizations only for compatible CUDA
        if device == "cuda":
            try:
                pipe.enable_attention_slicing()
                state.log("Attention slicing enabled", "info")
            except Exception:
                pass

            try:
                pipe.enable_vae_slicing()
                state.log("VAE slicing enabled", "info")
            except Exception:
                pass

            # Only enable xformers if flash attention is supported
            if CUDA_FLASH_ATTENTION_OK:
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    state.log("xFormers memory efficient attention enabled", "info")
                except Exception:
                    pass

        state.log("SD model loaded successfully", "success")
        print("SD model loaded successfully")
        state.sd_pipe = pipe

        # Also load CLIP model for similarity scoring (silently)
        try:
            load_clip_model(state, device=device)
            print("CLIP model pre-loaded successfully")
        except Exception as clip_e:
            # Log but don't fail - CLIP can be loaded later during analysis
            state.log(f"CLIP pre-load skipped: {clip_e}", "info")

        return pipe
    except Exception as e:
        state.log(f"Failed to load SD model: {str(e)}", "error")
        raise


def load_clip_model(
    state: "DashboardState",
    device: str = "cpu",
) -> Any:
    """Load CLIP model for similarity scoring.

    Args:
        state: Dashboard state to store the model
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded CLIPScore model
    """
    if state.clip_model is not None:
        return state.clip_model

    try:
        from torchmetrics.multimodal import CLIPScore

        clip_model = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
        clip_model = clip_model.to(device)
        state.clip_model = clip_model
        state.clip_device = device
        return clip_model
    except Exception as e:
        raise RuntimeError(f"Failed to load CLIP model: {e}") from e


def load_sae_model(
    state: "DashboardState",
    sae_model_id: str,
    sae_config: dict,
    get_sae_hyperparameters,
    get_feature_sums_path,
    get_model_path,
) -> Any:
    """Load SAE model with weights and feature sums from config.

    Args:
        state: Dashboard state to store the model
        sae_model_id: ID of the SAE model to load from config
        sae_config: SAE configuration dictionary
        get_sae_hyperparameters: Function to get hyperparameters
        get_feature_sums_path: Function to get feature sums path
        get_model_path: Function to get model path

    Returns:
        Loaded SAE model
    """
    try:
        from overcomplete.sae import TopKSAE
    except ImportError as e:
        state.log(f"SAE libraries not available: {e}", "error")
        raise

    if not sae_config:
        state.log("SAE config not loaded", "error")
        raise RuntimeError("SAE configuration not loaded")

    state.log("Loading SAE model...", "loading")

    try:
        # Get hyperparameters from config
        hyperparams = get_sae_hyperparameters(sae_config, sae_model_id)
        if not hyperparams:
            raise ValueError(f"No hyperparameters found for SAE model: {sae_model_id}")

        topk = hyperparams.get("topk", 32)
        nb_concepts = hyperparams.get("nb_concepts", 32768)

        state.log(f"SAE config: topk={topk}, nb_concepts={nb_concepts}", "info")

        # Get feature sums path
        feature_sums_path = get_feature_sums_path(sae_config, sae_model_id)
        if not feature_sums_path or not feature_sums_path.exists():
            raise FileNotFoundError(f"Feature sums not found: {feature_sums_path}")

        state.log(f"Loading feature sums from: {feature_sums_path.name}", "loading")

        # Load feature sums (concept statistics)
        device = "cuda" if (torch.cuda.is_available() and CUDA_COMPATIBLE) else "cpu"
        feature_sums = torch.load(feature_sums_path, map_location=device, weights_only=False)

        state.log(f"Feature sums loaded: {len(feature_sums)} concepts", "success")

        # Input shape for UNET_UP_1_ATT_1 layer
        input_shape = 1280

        # Get model weights path
        model_path = get_model_path(sae_config, sae_model_id)
        if not model_path or not model_path.exists():
            raise FileNotFoundError(f"SAE model weights not found: {model_path}")

        state.log(f"Loading SAE weights from: {model_path.name}", "loading")

        # Create SAE model
        state.log("Initializing SAE architecture...", "loading")
        sae = TopKSAE(
            input_shape=input_shape,
            nb_concepts=nb_concepts,
            top_k=topk,
            device=device,
        )

        # Load pre-trained weights
        model_weights = torch.load(model_path, map_location=device, weights_only=False)
        sae.load_state_dict(model_weights)

        sae = sae.to(device).to(dtype=torch.float32)
        sae.eval()

        state.log("SAE model loaded successfully", "success")
        state.sae_model = sae
        state.sae_stats = feature_sums
        state.current_sae_model_id = sae_model_id

        return sae
    except Exception as e:
        state.log(f"Failed to load SAE model: {str(e)}", "error")
        raise


def load_nudenet_model(state: "DashboardState") -> Any:
    """Load NudeNet detector.

    Args:
        state: Dashboard state to store the detector

    Returns:
        Loaded NudeDetector
    """
    state.log("Loading NudeNet detector...", "loading")

    try:
        from nudenet import NudeDetector
    except ImportError as e:
        error_msg = f"NudeNet not available: {e}"
        state.log(error_msg, "error")
        raise ImportError(error_msg)

    try:
        # Force CPU provider if flash attention is not supported
        if not CUDA_FLASH_ATTENTION_OK:
            state.log("Using CPU for NudeNet (Flash Attention not supported on this GPU)", "info")
            providers = ["CPUExecutionProvider"]
            detector = NudeDetector(providers=providers)
        else:
            detector = NudeDetector()

        state.log("NudeNet detector initialized", "info")
        state.nudenet_detector = detector
        state.log("NudeNet detector loaded successfully", "success")
        return detector
    except Exception as e:
        state.log(f"Failed to load NudeNet: {str(e)}", "error")
        raise
