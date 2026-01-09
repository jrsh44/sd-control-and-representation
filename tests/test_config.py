"""Unit tests for src/models/config.py"""

from src.models.config import ModelConfig, ModelRegistry


def test_model_config_creation():
    """Test ModelConfig named tuple creation."""
    config = ModelConfig(
        model_id="test/model",
        source="huggingface",
        name="test_model",
        pipeline_type="sd_v1_5",
    )
    assert config.model_id == "test/model"
    assert config.source == "huggingface"
    assert config.name == "test_model"
    assert config.pipeline_type == "sd_v1_5"


def test_model_config_default_pipeline():
    """Test ModelConfig default pipeline_type."""
    config = ModelConfig(
        model_id="test/model",
        source="huggingface",
        name="test_model",
    )
    assert config.pipeline_type == "sd_v1_5"


def test_model_registry_sd_v1_5():
    """Test SD_V1_5 model in registry."""
    model = ModelRegistry.SD_V1_5
    assert model.model_id == "sd-legacy/stable-diffusion-v1-5"
    assert model.source == "huggingface"
    assert model.config_name == "sd_v1_5"
    assert model.pipeline_type == "sd_v1_5"


def test_model_registry_finetuned_saeuron():
    """Test FINETUNED_SAEURON model in registry."""
    model = ModelRegistry.FINETUNED_SAEURON
    assert "drive.google.com" in model.model_id
    assert model.source == "gdrive"
    assert model.config_name == "finetuned_sd_saeuron"
    assert model.pipeline_type == "sd_v1_5"


def test_model_registry_sd_v3():
    """Test SD_V3 model in registry."""
    model = ModelRegistry.SD_V3
    assert model.model_id == "stabilityai/stable-diffusion-3-medium-diffusers"
    assert model.source == "huggingface"
    assert model.config_name == "sd_v3"
    assert model.pipeline_type == "sd_v3"


def test_model_registry_enum_properties():
    """Test that all ModelRegistry enums have required properties."""
    for model in ModelRegistry:
        assert isinstance(model.model_id, str)
        assert isinstance(model.source, str)
        assert isinstance(model.config_name, str)
        assert model.pipeline_type in ["sd_v1_5", "sd_v3"]


def test_model_registry_unique_names():
    """Test that all models have unique config names."""
    names = [model.config_name for model in ModelRegistry]
    assert len(names) == len(set(names))


def test_model_registry_value_access():
    """Test accessing ModelConfig through value attribute."""
    sd_v1_5 = ModelRegistry.SD_V1_5
    assert isinstance(sd_v1_5.value, ModelConfig)
    assert sd_v1_5.value.model_id == sd_v1_5.model_id
