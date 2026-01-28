"""
Enums for model IDs and their configurations.
"""

from enum import Enum
from typing import Literal, NamedTuple


class ModelConfig(NamedTuple):
    """Configuration for a model including its ID and source."""

    model_id: str
    source: str
    name: str
    pipeline_type: Literal["sd_v1_5", "sd_v3"] = "sd_v1_5"


class ModelRegistry(Enum):
    """Enum containing available model configurations."""

    SD_V1_5 = ModelConfig(
        model_id="sd-legacy/stable-diffusion-v1-5",
        source="huggingface",
        name="sd_v1_5",
        pipeline_type="sd_v1_5",
    )

    FINETUNED_SAEURON = ModelConfig(
        model_id="https://drive.google.com/drive/folders/14_ckUo_JLOt8opkIXPVmjk6nSuKjFm-C",
        source="gdrive",
        name="finetuned_sd_saeuron",
        pipeline_type="sd_v1_5",
    )

    SD_V3 = ModelConfig(
        model_id="stabilityai/stable-diffusion-3-medium-diffusers",
        source="huggingface",
        name="sd_v3",
        pipeline_type="sd_v3",
    )

    @property
    def model_id(self) -> str:
        return self.value.model_id

    @property
    def source(self) -> str:
        return self.value.source

    @property
    def config_name(self) -> str:
        return self.value.name

    @property
    def pipeline_type(self) -> Literal["sd_v1_5", "sd_v3"]:
        return self.value.pipeline_type
