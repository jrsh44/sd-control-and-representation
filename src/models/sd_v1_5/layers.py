"""
Layer definitions for Stable Diffusion v1.5.
Defines all available layers for representation capture.
"""

from enum import Enum


class LayerPath(Enum):
    """
    Comprehensive paths to all modules in Stable Diffusion 1.5.
    """

    # ============================================================================
    # CLIP Text Encoder
    # ============================================================================

    # Shape: [batch, 77 tokens, 768 dims]
    TEXT_TOKEN_EMBEDS = "text_encoder.text_model.embeddings.token_embedding"  # noqa: S105
    TEXT_EMBEDDING_MID = "text_encoder.text_model.encoder.layers.5"
    TEXT_EMBEDDING_FINAL = "text_encoder.text_model.encoder.layers.11"

    # ============================================================================
    # U-Net
    # ============================================================================

    # Shape: [batch, 1280 dims]
    UNET_TIME_EMBED = "unet.time_embedding"

    # Shape: [batch, 320 channels, 64, 64] for 512x512 images
    UNET_INPUT_LATENT = "unet.conv_in"

    # Shape: [batch, 4 channels, 64, 64]
    UNET_OUTPUT_PREDICTED_NOISE = "unet.conv_out"

    # Shape: [batch, 320 channels, 64, 64]
    UNET_DOWN_0_RES_0 = "unet.down_blocks.0.resnets.0"
    UNET_DOWN_0_RES_1 = "unet.down_blocks.0.resnets.1"
    UNET_DOWN_0_DOWNSAMPLE = "unet.down_blocks.0.downsamplers.0"

    # Shape: [batch, 640 channels, 32, 32]
    UNET_DOWN_1_RES_0 = "unet.down_blocks.1.resnets.0"
    UNET_DOWN_1_ATT_0 = "unet.down_blocks.1.attentions.0.transformer_blocks.0"
    UNET_DOWN_1_RES_1 = "unet.down_blocks.1.resnets.1"
    UNET_DOWN_1_ATT_1 = "unet.down_blocks.1.attentions.1.transformer_blocks.0"
    UNET_DOWN_1_DOWNSAMPLE = "unet.down_blocks.1.downsamplers.0"

    # Shape: [batch, 1280 channels, 16, 16]
    UNET_DOWN_2_RES_0 = "unet.down_blocks.2.resnets.0"
    UNET_DOWN_2_ATT_0 = "unet.down_blocks.2.attentions.0.transformer_blocks.0"
    UNET_DOWN_2_RES_1 = "unet.down_blocks.2.resnets.1"
    UNET_DOWN_2_ATT_1 = "unet.down_blocks.2.attentions.1.transformer_blocks.0"
    UNET_DOWN_2_DOWNSAMPLE = "unet.down_blocks.2.downsamplers.0"

    # Shape: [batch, 1280 channels, 8, 8]
    UNET_DOWN_3_RES_0 = "unet.down_blocks.3.resnets.0"
    UNET_DOWN_3_ATT_0 = "unet.down_blocks.3.attentions.0.transformer_blocks.0"
    UNET_DOWN_3_RES_1 = "unet.down_blocks.3.resnets.1"
    UNET_DOWN_3_ATT_1 = "unet.down_blocks.3.attentions.1.transformer_blocks.0"

    # Shape: [batch, 1280 channels, 8, 8]
    UNET_MID_RES_0 = "unet.mid_block.resnets.0"
    UNET_MID_ATT = "unet.mid_block.attentions.0.transformer_blocks.0"
    UNET_MID_RES_1 = "unet.mid_block.resnets.1"

    # Shape: [batch, 1280 channels, 8, 8]
    UNET_UP_3_RES_0 = "unet.up_blocks.3.resnets.0"
    UNET_UP_3_ATT_0 = "unet.up_blocks.3.attentions.0.transformer_blocks.0"
    UNET_UP_3_RES_1 = "unet.up_blocks.3.resnets.1"
    UNET_UP_3_ATT_1 = "unet.up_blocks.3.attentions.1.transformer_blocks.0"
    UNET_UP_3_RES_2 = "unet.up_blocks.3.resnets.2"
    UNET_UP_3_ATT_2 = "unet.up_blocks.3.attentions.2.transformer_blocks.0"

    # Shape: [batch, 1280 channels, 16, 16]
    UNET_UP_2_RES_0 = "unet.up_blocks.2.resnets.0"
    UNET_UP_2_ATT_0 = "unet.up_blocks.2.attentions.0.transformer_blocks.0"
    UNET_UP_2_RES_1 = "unet.up_blocks.2.resnets.1"
    UNET_UP_2_ATT_1 = "unet.up_blocks.2.attentions.1.transformer_blocks.0"
    UNET_UP_2_RES_2 = "unet.up_blocks.2.resnets.2"
    UNET_UP_2_ATT_2 = "unet.up_blocks.2.attentions.2.transformer_blocks.0"

    # Shape: [batch, 640 channels, 32, 32]
    UNET_UP_1_RES_0 = "unet.up_blocks.1.resnets.0"
    UNET_UP_1_ATT_0 = "unet.up_blocks.1.attentions.0.transformer_blocks.0"
    UNET_UP_1_RES_1 = "unet.up_blocks.1.resnets.1"
    UNET_UP_1_ATT_1 = "unet.up_blocks.1.attentions.1.transformer_blocks.0"
    UNET_UP_1_RES_2 = "unet.up_blocks.1.resnets.2"
    UNET_UP_1_ATT_2 = "unet.up_blocks.1.attentions.2.transformer_blocks.0"

    # Shape: [batch, 320 channels, 64, 64]
    UNET_UP_0_RES_0 = "unet.up_blocks.0.resnets.0"
    UNET_UP_0_RES_1 = "unet.up_blocks.0.resnets.1"
    UNET_UP_0_RES_2 = "unet.up_blocks.0.resnets.2"

    # ============================================================================
    # VAE
    # ============================================================================

    # Shape: [batch, 3 channels, 512, 512]
    VAE_ENCODER_INPUT = "vae.encoder.conv_in"

    # Shape: [batch, 3 channels, 512, 512]
    VAE_DECODER_OUTPUT = "vae.decoder.conv_out"

    def __str__(self):
        # Allows using enum instance as raw path string
        return self.value
