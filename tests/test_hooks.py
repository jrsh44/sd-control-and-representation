"""Unit tests for src/models/sd_v1_5/hooks.py and layers.py"""

import pytest

try:
    import torch
    from src.models.sd_v1_5.hooks import get_nested_module

    TORCH_AVAILABLE = True
except (ImportError, AttributeError):
    TORCH_AVAILABLE = False
    torch = None

# Import LayerPath separately since it doesn't require torch to be working
try:
    from src.models.sd_v1_5.layers import LayerPath

    LAYERPATH_AVAILABLE = True
except ImportError:
    LAYERPATH_AVAILABLE = False


# =============================================================================
# LayerPath Tests
# =============================================================================


@pytest.mark.skipif(not LAYERPATH_AVAILABLE, reason="LayerPath not available")
def test_layer_path_enum():
    """Test LayerPath enum has expected values."""
    assert hasattr(LayerPath, "TEXT_TOKEN_EMBEDS")
    assert hasattr(LayerPath, "UNET_UP_1_ATT_1")
    assert len(list(LayerPath)) > 0


# =============================================================================
# get_nested_module Tests
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_nested_module_single_level():
    """Test retrieving a single-level nested module."""

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(10, 10)

    model = MockModel()
    result = get_nested_module(model, "layer1")

    assert isinstance(result, torch.nn.Linear)
    assert result.in_features == 10


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_nested_module_multi_level():
    """Test retrieving a multi-level nested module."""

    class MockSubModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, kernel_size=3)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = MockSubModule()

    model = MockModel()
    result = get_nested_module(model, "encoder.conv")

    assert isinstance(result, torch.nn.Conv2d)
    assert result.in_channels == 3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_nested_module_deep_nesting():
    """Test retrieving a deeply nested module."""

    class Level3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.final = torch.nn.ReLU()

    class Level2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.level3 = Level3()

    class Level1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.level2 = Level2()

    model = Level1()
    result = get_nested_module(model, "level2.level3.final")

    assert isinstance(result, torch.nn.ReLU)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_nested_module_with_sequential():
    """Test retrieving module from Sequential container using index."""

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 10),
            )

    model = MockModel()

    # Access by index
    result = get_nested_module(model, "layers.0")
    assert isinstance(result, torch.nn.Linear)
    assert result.in_features == 10

    result = get_nested_module(model, "layers.1")
    assert isinstance(result, torch.nn.ReLU)

    result = get_nested_module(model, "layers.2")
    assert isinstance(result, torch.nn.Linear)
    assert result.out_features == 10


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_nested_module_with_modulelist():
    """Test retrieving module from ModuleList using index."""

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [
                    torch.nn.Linear(10, 10),
                    torch.nn.Linear(10, 10),
                ]
            )

    model = MockModel()

    result = get_nested_module(model, "blocks.0")
    assert isinstance(result, torch.nn.Linear)

    result = get_nested_module(model, "blocks.1")
    assert isinstance(result, torch.nn.Linear)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_nested_module_negative_index():
    """Test retrieving module using negative index."""

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 10),
            )

    model = MockModel()

    # Access last layer with -1
    result = get_nested_module(model, "layers.-1")
    assert isinstance(result, torch.nn.Linear)
    assert result.out_features == 10


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_nested_module_complex_path():
    """Test retrieving module with complex nested path including indices."""

    class InnerBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ops = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3),
                torch.nn.BatchNorm2d(64),
            )

    class OuterBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [
                    InnerBlock(),
                    InnerBlock(),
                ]
            )

    model = OuterBlock()

    # Navigate: blocks -> [0] -> ops -> [1]
    result = get_nested_module(model, "blocks.0.ops.1")
    assert isinstance(result, torch.nn.BatchNorm2d)
    assert result.num_features == 64


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_nested_module_invalid_path():
    """Test that invalid path raises AttributeError."""

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(10, 10)

    model = MockModel()

    with pytest.raises(AttributeError):
        get_nested_module(model, "nonexistent_layer")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_nested_module_invalid_index():
    """Test that invalid index raises IndexError."""

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.ReLU(),
            )

    model = MockModel()

    with pytest.raises(IndexError):
        get_nested_module(model, "layers.10")  # Only 2 layers exist


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_nested_module_empty_path():
    """Test that empty path returns the model itself."""

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

    model = MockModel()
    result = get_nested_module(model, "")

    assert result is model


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_nested_module_parametric():
    """Test with various module types."""

    class ComplexModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, kernel_size=3)
            self.bn = torch.nn.BatchNorm2d(64)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(0.5)
            self.fc = torch.nn.Linear(64, 10)

    model = ComplexModel()

    # Test each type
    assert isinstance(get_nested_module(model, "conv"), torch.nn.Conv2d)
    assert isinstance(get_nested_module(model, "bn"), torch.nn.BatchNorm2d)
    assert isinstance(get_nested_module(model, "relu"), torch.nn.ReLU)
    assert isinstance(get_nested_module(model, "dropout"), torch.nn.Dropout)
    assert isinstance(get_nested_module(model, "fc"), torch.nn.Linear)
