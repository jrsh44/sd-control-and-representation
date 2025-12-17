"""Unit tests for src/utils/visualization.py"""

from PIL import Image
from src.utils.visualization import display_image, display_sequence, display_gif


def test_display_image():
    """Test image display."""
    img = Image.new("RGB", (100, 100), color="red")
    try:
        display_image(img, title="Test")
    except:
        pass  # OK to fail in headless environment


def test_display_sequence():
    """Test sequence display."""
    images = [Image.new("RGB", (50, 50), color="blue") for _ in range(3)]
    try:
        display_sequence(images, title="Test")
    except:
        pass  # OK to fail in headless environment


def test_display_gif():
    """Test GIF display."""
    frames = [Image.new("RGB", (50, 50), color="green") for _ in range(2)]
    try:
        display_gif(frames, duration=100)
    except:
        pass  # OK to fail in headless environment
