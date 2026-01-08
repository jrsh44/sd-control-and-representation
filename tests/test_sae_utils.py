"""Unit tests for src/models/sae/training/utils.py"""

import pytest

try:
    import torch
    from src.models.sae.training.utils import (
        create_warmup_cosine_scheduler,
        extract_input,
        get_dictionary,
    )
    from src.models.sae.training.config import SchedulerConfig

    TORCH_AVAILABLE = True
except (ImportError, AttributeError):
    TORCH_AVAILABLE = False
    torch = None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_extract_input_tuple():
    """Test extracting input from tuple."""
    data = torch.randn(4, 10)
    labels = torch.randint(0, 5, (4,))
    result = extract_input((data, labels))
    assert torch.equal(result, data)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_extract_input_list():
    """Test extracting input from list."""
    data = torch.randn(4, 10)
    labels = torch.randint(0, 5, (4,))
    result = extract_input([data, labels])
    assert torch.equal(result, data)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_extract_input_dict_with_data():
    """Test extracting input from dict with 'data' key."""
    data = torch.randn(4, 10)
    result = extract_input({"data": data, "label": torch.tensor([1])})
    assert torch.equal(result, data)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_extract_input_dict_without_data():
    """Test extracting input from dict without 'data' key."""
    result = extract_input({"wrong_key": torch.randn(4, 10)})
    assert result is None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_extract_input_tensor():
    """Test extracting input from tensor directly."""
    data = torch.randn(4, 10)
    result = extract_input(data)
    assert torch.equal(result, data)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_dictionary_property():
    """Test get_dictionary with property access."""

    class MockModel:
        @property
        def get_dictionary(self):
            return torch.randn(10, 64)

    model = MockModel()
    dictionary = get_dictionary(model)
    assert isinstance(dictionary, torch.Tensor)
    assert dictionary.shape == (10, 64)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_get_dictionary_method():
    """Test get_dictionary with method access."""

    class MockModel:
        def get_dictionary(self):
            return torch.randn(10, 64)

    model = MockModel()
    dictionary = get_dictionary(model)
    assert isinstance(dictionary, torch.Tensor)
    assert dictionary.shape == (10, 64)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_create_warmup_cosine_scheduler_disabled():
    """Test scheduler creation when disabled."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    config = SchedulerConfig(enabled=False)

    scheduler = create_warmup_cosine_scheduler(optimizer, config, total_steps=1000)
    assert scheduler is None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_create_warmup_cosine_scheduler_no_warmup():
    """Test scheduler creation with no warmup steps."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    config = SchedulerConfig(enabled=True, warmup_steps=0)

    scheduler = create_warmup_cosine_scheduler(optimizer, config, total_steps=1000)
    assert scheduler is None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_create_warmup_cosine_scheduler_valid():
    """Test scheduler creation with valid config."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    config = SchedulerConfig(
        enabled=True,
        warmup_steps=100,
        warmup_start_factor=0.1,
        min_lr_ratio=0.01,
    )

    scheduler = create_warmup_cosine_scheduler(optimizer, config, total_steps=1000)
    assert scheduler is not None

    # Test that learning rate changes over steps
    initial_lr = optimizer.param_groups[0]["lr"]

    # Step through warmup
    for _ in range(50):
        scheduler.step()
    warmup_lr = optimizer.param_groups[0]["lr"]

    # LR should increase during warmup
    assert warmup_lr > initial_lr * config.warmup_start_factor


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_create_warmup_cosine_scheduler_warmup_too_large():
    """Test scheduler when warmup_steps exceeds total_steps."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    config = SchedulerConfig(enabled=True, warmup_steps=2000)

    # Should clamp warmup_steps and still create scheduler
    scheduler = create_warmup_cosine_scheduler(optimizer, config, total_steps=1000)
    assert scheduler is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_create_warmup_cosine_scheduler_min_lr_ratio():
    """Test that scheduler respects min_lr_ratio."""
    model = torch.nn.Linear(10, 10)
    base_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    config = SchedulerConfig(
        enabled=True,
        warmup_steps=10,
        min_lr_ratio=0.1,
    )

    scheduler = create_warmup_cosine_scheduler(optimizer, config, total_steps=100)

    # Step to the end
    for _ in range(100):
        scheduler.step()

    final_lr = optimizer.param_groups[0]["lr"]
    # Final LR should be close to base_lr * min_lr_ratio
    assert final_lr >= base_lr * config.min_lr_ratio * 0.9  # Allow 10% tolerance
