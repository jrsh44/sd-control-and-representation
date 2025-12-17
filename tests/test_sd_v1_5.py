"""Unit tests for src/models/sd_v1_5/layers.py and hooks.py"""

import pytest

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from src.models.sd_v1_5.hooks import get_nested_module
from src.models.sd_v1_5.layers import LayerPath


def test_layer_path_enum():
    """Test LayerPath enum has expected values."""
    assert hasattr(LayerPath, "TEXT_TOKEN_EMBEDS")
    assert hasattr(LayerPath, "UNET_UP_1_ATT_1")
    assert len(list(LayerPath)) > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_nested_module_basic():
    """Test basic nested module access."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
    result = get_nested_module(model, "0")
    assert isinstance(result, nn.Linear)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_nested_module_deep():
    """Test deep nested module access."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.ModuleDict({"layer1": nn.Linear(10, 20)})

    model = Model()
    result = get_nested_module(model, "encoder.layer1")
    assert isinstance(result, nn.Linear)
