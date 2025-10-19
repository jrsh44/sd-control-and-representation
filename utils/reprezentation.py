import torch
from diffusers import StableDiffusionPipeline
from typing import List, Dict, Any
from enum import Enum

# --- 1. LayerPath: Simple Enum for Layer Paths ---


class LayerPath(Enum):
    """
    Comprehensive paths to all meaningful modules in Stable Diffusion 1.5.
    Each path points to a specific layer for capturing intermediate representations.
    """

    # ============================================================================
    # A. CLIP Text Encoder (Conditioning)
    # Converts prompt tokens into semantic embeddings that guide image generation
    # ============================================================================

    # Raw Token Embeddings - BASELINE
    # Direct lookup from CLIP's embedding table, before any contextual processing.
    # Shape: [batch, 77 tokens, 768 dims]
    # Usage: Baseline representation before transformer processing
    # Analysis: Shows initial token meanings before context is applied
    TEXT_TOKEN_EMBEDS = "text_encoder.text_model.embeddings.token_embedding"

    # Middle CLIP Layer (Layer 5/11) - MID-LEVEL SEMANTICS
    # Intermediate representation halfway through CLIP's transformer stack.
    # Shape: [batch, 77 tokens, 768 dims]
    # Usage: Analyzing early/mid-level semantic understanding evolution
    # Analysis: Less redundant than every layer, shows semantic refinement
    TEXT_EMBEDDING_MID = "text_encoder.text_model.encoder.layers.5"

    # Final Text Embedding (Layer 11/11) - MOST IMPORTANT FOR CONDITIONING
    # Output of the last CLIP transformer layer.
    # Shape: [batch, 77 tokens, 768 dims]
    # Usage: Serves as Key (K) and Value (V) in U-Net cross-attention
    # Analysis: This is what U-Net "reads" to understand what to generate
    TEXT_EMBEDDING_FINAL = "text_encoder.text_model.encoder.layers.11"

    # ============================================================================
    # B. U-Net (Latent Visual Features and Cross-Attention)
    # Iteratively denoises latent representations using text guidance
    # ============================================================================

    # --- Time Conditioning and Input/Output ---

    # Time Embedding Vector - TIMESTEP CONDITIONING
    # Encodes the current denoising timestep (e.g., step 20/50).
    # Shape: [batch, 1280 dims]
    # Usage: Injected into ResNet blocks to condition on denoising progress
    # Analysis: Compare across timesteps; early steps = composition, late = details
    UNET_TIME_EMBED = "unet.time_embedding"

    # Noisy Latent Image (After Input Convolution) - U-NET INPUT
    # Initial latent after first convolution layer.
    # Shape: [batch, 320 channels, 64, 64] for 512x512 images
    # Usage: Starting point of U-Net processing for each denoising step
    # Analysis: Track how noise evolves across timesteps
    UNET_INPUT_LATENT = "unet.conv_in"

    # Predicted Noise (U-Net Output) - KEY FOR DENOISING
    # The noise prediction subtracted from latent in each step.
    # Shape: [batch, 4 channels, 64, 64]
    # Usage: Core model output used by scheduler for denoising
    # Analysis: Visualize predicted noise patterns, track accuracy
    UNET_OUTPUT_PREDICTED_NOISE = "unet.conv_out"

    # --- Down Blocks (Feature Hierarchy and Resolution Reduction) ---

    # Down Block 0 (64x64) - HIGHEST RESOLUTION, LOCAL DETAILS
    # First downsampling stage, no cross-attention yet.
    # Shape: [batch, 320 channels, 64, 64]
    # Usage: Local feature extraction, fine-grained spatial information
    # Analysis: Pure visual features without text conditioning
    # ⭐ MOST USEFUL: RES_1 (final features before downsampling)
    UNET_DOWN_0_RES_0 = "unet.down_blocks.0.resnets.0"  # Initial local features
    UNET_DOWN_0_RES_1 = (
        "unet.down_blocks.0.resnets.1"  # ⭐ Refined features before downsample
    )
    UNET_DOWN_0_DOWNSAMPLE = (
        "unet.down_blocks.0.downsamplers.0"  # After resolution reduction
    )

    # Down Block 1 (32x32) - MID/HIGH RESOLUTION, CROSS-ATTENTION STARTS
    # First block with cross-attention, text conditioning begins.
    # Shape: [batch, 640 channels, 32, 32]
    # Usage: Local features start being conditioned by text semantics
    # Analysis: Compare ResNet vs Attention outputs to see text's impact
    # ⭐ MOST USEFUL: ATT_1 (final text-conditioned features) or pair RES_1 + ATT_1
    UNET_DOWN_1_RES_0 = "unet.down_blocks.1.resnets.0"  # Visual features before text
    UNET_DOWN_1_ATT_0 = "unet.down_blocks.1.attentions.0.transformer_blocks.0"  # First text conditioning
    UNET_DOWN_1_RES_1 = "unet.down_blocks.1.resnets.1"  # Further visual refinement
    UNET_DOWN_1_ATT_1 = "unet.down_blocks.1.attentions.1.transformer_blocks.0"  # ⭐ Final conditioned features
    UNET_DOWN_1_DOWNSAMPLE = (
        "unet.down_blocks.1.downsamplers.0"  # After resolution reduction
    )

    # Down Block 2 (16x16) - LOWER RESOLUTION, GLOBAL FEATURES
    # More abstract features, stronger semantic alignment.
    # Shape: [batch, 1280 channels, 16, 16]
    # Usage: Mid-level semantic features, object-level understanding
    # Analysis: See how objects are represented at coarser scales
    # ⭐ MOST USEFUL: ATT_0 or ATT_1 (strong text-visual alignment for objects)
    UNET_DOWN_2_RES_0 = "unet.down_blocks.2.resnets.0"  # Visual features before text
    UNET_DOWN_2_ATT_0 = "unet.down_blocks.2.attentions.0.transformer_blocks.0"  # ⭐ Object-level alignment
    UNET_DOWN_2_RES_1 = "unet.down_blocks.2.resnets.1"  # Further visual refinement
    UNET_DOWN_2_ATT_1 = "unet.down_blocks.2.attentions.1.transformer_blocks.0"  # ⭐ Refined semantic features
    UNET_DOWN_2_DOWNSAMPLE = (
        "unet.down_blocks.2.downsamplers.0"  # After resolution reduction
    )

    # Down Block 3 (8x8) - LOWEST RESOLUTION BEFORE MID, VERY ABSTRACT
    # Deepest downsampling block, most abstract visual features.
    # Shape: [batch, 1280 channels, 8, 8]
    # Usage: High-level semantic concepts, global scene structure
    # Analysis: Most abstract visual representations before mid block
    # Note: Last down block has no downsampler
    # ⭐ MOST USEFUL: ATT_1 (highest abstraction before bottleneck)
    UNET_DOWN_3_RES_0 = "unet.down_blocks.3.resnets.0"  # Visual features before text
    UNET_DOWN_3_ATT_0 = "unet.down_blocks.3.attentions.0.transformer_blocks.0"  # Global semantic alignment
    UNET_DOWN_3_RES_1 = "unet.down_blocks.3.resnets.1"  # Further visual refinement
    UNET_DOWN_3_ATT_1 = "unet.down_blocks.3.attentions.1.transformer_blocks.0"  # ⭐ Most abstract features

    # --- Mid Block (Bottleneck: Lowest Resolution, Maximum Context) ---
    # ⭐⭐⭐ CRITICAL BLOCK: Determines overall composition and structure

    # Mid Block ResNet 0 - BEFORE ATTENTION
    # Local processing at the bottleneck.
    # Shape: [batch, 1280 channels, 8, 8]
    # Usage: Feature refinement before cross-attention
    # Analysis: Pure visual features at maximum abstraction level
    UNET_MID_RES_0 = "unet.mid_block.resnets.0"  # Visual features before text

    # Mid Block Attention - ⭐⭐⭐ SINGLE MOST IMPORTANT LAYER
    # Deepest point in U-Net, highest semantic level.
    # Shape: [batch, 1280 channels, 8, 8]
    # Usage: Global scene composition, high-level semantic alignment
    # Analysis: QK^T maps show which words influence which regions most
    # Tip: This layer determines overall image structure and object placement
    # Why critical: Only one attention layer here, maximum receptive field
    UNET_MID_ATT = (
        "unet.mid_block.attentions.0.transformer_blocks.0"  # ⭐⭐⭐ MOST IMPORTANT
    )

    # Mid Block ResNet 1 - AFTER ATTENTION
    # Feature refinement after cross-attention.
    # Shape: [batch, 1280 channels, 8, 8]
    # Usage: Processes text-conditioned features before upsampling
    # Analysis: Compare with MID_RES_0 to see attention's effect
    # Tip: Captures (RES_0 → ATT → RES_1) shows full text integration
    UNET_MID_RES_1 = "unet.mid_block.resnets.1"  # ⭐ Post-attention features

    # --- Up Blocks (Resolution Recovery and Detail Restoration) ---

    # Up Block 3 (8x8→16x16) - START OF UPSAMPLING, ABSTRACT DETAILS
    # First upsampling stage, begins detail recovery.
    # Shape: [batch, 1280 channels, 8, 8] → [16, 16] after upsample
    # Usage: Recovers spatial structure from abstract representations
    # Analysis: See how abstract concepts translate to spatial features
    # ⭐ MOST USEFUL: ATT_2 (final features before spatial expansion) or sequence ATT_0→ATT_1→ATT_2
    # Why 3 ResNets: Skip connections from encoder create 3 refinement stages
    UNET_UP_3_RES_0 = "unet.up_blocks.3.resnets.0"  # First refinement after mid
    UNET_UP_3_ATT_0 = (
        "unet.up_blocks.3.attentions.0.transformer_blocks.0"  # Initial text alignment
    )
    UNET_UP_3_RES_1 = "unet.up_blocks.3.resnets.1"  # Second refinement
    UNET_UP_3_ATT_1 = (
        "unet.up_blocks.3.attentions.1.transformer_blocks.0"  # Further alignment
    )
    UNET_UP_3_RES_2 = "unet.up_blocks.3.resnets.2"  # Final refinement
    UNET_UP_3_ATT_2 = "unet.up_blocks.3.attentions.2.transformer_blocks.0"  # ⭐ Most refined abstract features

    # Up Block 2 (16x16→32x32) - STRUCTURE RECOVERY
    # Mid-level detail restoration, object shapes emerge.
    # Shape: [batch, 1280 channels, 16, 16] → [32, 32]
    # Usage: Refines object boundaries and spatial relationships
    # Analysis: See how global structure becomes more defined
    # ⭐ MOST USEFUL: ATT_1 or ATT_2 (where object boundaries solidify)
    # Why 3 layers: Each processes different skip connection information
    UNET_UP_2_RES_0 = "unet.up_blocks.2.resnets.0"  # Initial structure recovery
    UNET_UP_2_ATT_0 = (
        "unet.up_blocks.2.attentions.0.transformer_blocks.0"  # Early boundary alignment
    )
    UNET_UP_2_RES_1 = "unet.up_blocks.2.resnets.1"  # Boundary refinement
    UNET_UP_2_ATT_1 = "unet.up_blocks.2.attentions.1.transformer_blocks.0"  # ⭐ Object shape definition
    UNET_UP_2_RES_2 = "unet.up_blocks.2.resnets.2"  # Final structure polish
    UNET_UP_2_ATT_2 = "unet.up_blocks.2.attentions.2.transformer_blocks.0"  # ⭐ Refined spatial structure

    # Up Block 1 (32x32→64x64) - FINE DETAIL RECOVERY
    # High-resolution detail synthesis, textures and edges.
    # Shape: [batch, 640 channels, 32, 32] → [64, 64]
    # Usage: Synthesizes fine-grained details, textures, precise boundaries
    # Analysis: See precise spatial alignment between text and visual features
    # Tip: This layer handles texture, fine details, object boundaries
    # ⭐ MOST USEFUL: ATT_2 (finest text-aligned details) or comparing ATT_0→ATT_2 progression
    # Why monitor multiple: Shows how fine details emerge progressively
    UNET_UP_1_RES_0 = "unet.up_blocks.1.resnets.0"  # Initial detail synthesis
    UNET_UP_1_ATT_0 = (
        "unet.up_blocks.1.attentions.0.transformer_blocks.0"  # Coarse detail alignment
    )
    UNET_UP_1_RES_1 = "unet.up_blocks.1.resnets.1"  # Detail refinement
    UNET_UP_1_ATT_1 = "unet.up_blocks.1.attentions.1.transformer_blocks.0"  # ⭐ Texture/edge alignment
    UNET_UP_1_RES_2 = "unet.up_blocks.1.resnets.2"  # Final detail polish
    UNET_UP_1_ATT_2 = "unet.up_blocks.1.attentions.2.transformer_blocks.0"  # ⭐ Finest conditioned details

    # Up Block 0 (64x64) - HIGHEST RESOLUTION, FINAL DETAILS BEFORE VAE
    # Final refinement at full latent resolution, no cross-attention.
    # Shape: [batch, 320 channels, 64, 64]
    # Usage: Final polishing of visual features before VAE decoder
    # Analysis: See final latent representation before pixel conversion
    # Note: No cross-attention in this block (text influence already applied)
    # ⭐ MOST USEFUL: RES_2 (final U-Net features before output conv)
    # Why 3 ResNets: Progressive refinement of skip connections without text
    UNET_UP_0_RES_0 = "unet.up_blocks.0.resnets.0"  # Initial high-res refinement
    UNET_UP_0_RES_1 = "unet.up_blocks.0.resnets.1"  # Mid-stage polish
    UNET_UP_0_RES_2 = "unet.up_blocks.0.resnets.2"  # ⭐ Final features before conv_out

    # ============================================================================
    # C. VAE (Variational AutoEncoder)
    # Compresses/decompresses between pixel and latent space (8x per dimension)
    # ============================================================================

    # VAE Encoder Input - RAW PIXEL SPACE
    # First layer of VAE encoder, receives raw RGB pixels.
    # Shape: [batch, 3 channels, 512, 512] for input images
    # Usage: Starting point for compression to latent space
    # Analysis: Compare with encoded latents to understand information loss
    VAE_ENCODER_INPUT = "vae.encoder.conv_in"

    # VAE Decoder Output - RECONSTRUCTED PIXELS
    # Final layer of VAE decoder, produces RGB image from latents.
    # Shape: [batch, 3 channels, 512, 512] for output images
    # Usage: Final reconstruction back to pixel space
    # Analysis: The actual image you see, after latent decompression
    VAE_DECODER_OUTPUT = "vae.decoder.conv_out"

    def __str__(self):
        # Allows using enum instance as raw path string (value)
        return self.value


# --- 2. HELPER FUNCTION: Navigate Module Hierarchy ---


def get_nested_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Retrieves a nested module from a model using a dot-separated path string.

    Args:
        model (torch.nn.Module): The root model to navigate.
        path (str): Dot-separated path to the target module (e.g., "unet.down_blocks.1.resnets.0").

    Returns:
        torch.nn.Module: The target module at the specified path.
    """
    path_segments = path.split(".")
    current_module = model
    for name in path_segments:
        if name.isdigit() or name.startswith("-") and name[1:].isdigit():
            current_module = current_module[int(name)]
        else:
            current_module = getattr(current_module, name)

    return current_module


# --- 3. MAIN FUNCTION: Capture Layer Representations ---


def capture_layer_representations(
    pipe: StableDiffusionPipeline,
    prompt: str,
    layer_paths: List[LayerPath],
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: torch.Generator = None,
) -> List[torch.Tensor]:
    """
    Captures intermediate representations from specified layers during a single SD generation.
    Uses forward hooks to intercept and store activations from target layers.

    Args:
        pipe (StableDiffusionPipeline): The Stable Diffusion pipeline instance.
        prompt (str): Text prompt for image generation.
        layer_paths (List[LayerPath]): List of layer paths to capture representations from.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Classifier-free guidance scale.
        generator (torch.Generator): Random generator for reproducibility.

    Returns:
        List[torch.Tensor]: List of captured activation tensors, in the same order as layer_paths.
    """
    captured_representations: Dict[str, torch.Tensor] = {}
    hook_handles: List[Any] = []

    def create_capture_hook(name: str):
        """Factory function to create a hook that captures module output."""

        def hook(model, input, output):
            if isinstance(output, tuple):
                captured_representations[name] = output[0].detach().cpu()
            else:
                captured_representations[name] = output.detach().cpu()

        return hook

    # Register forward hooks on target layers
    for i, layer_enum in enumerate(layer_paths):
        path = str(layer_enum)
        hook_name = f"hook_{i}"

        try:
            module = get_nested_module(pipe, path)
            handle = module.register_forward_hook(create_capture_hook(hook_name))
            hook_handles.append(handle)
        except Exception as e:
            print(f"ERROR: Cannot find module for path '{path}'. Error: {e}")

    # Generate image to trigger activations
    pipe_args = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }

    pipe(**pipe_args)

    # Clean up hooks
    for handle in hook_handles:
        handle.remove()

    # Return list of tensors in the order specified by layer_paths
    results = [
        captured_representations[f"hook_{i}"]
        for i in range(len(layer_paths))
        if f"hook_{i}" in captured_representations
    ]

    return results
