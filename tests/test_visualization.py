"""Unit tests for src/utils/visualization.py"""

import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for tests

from unittest.mock import patch, MagicMock
from PIL import Image
import pytest
from src.utils.visualization import display_image, display_sequence, display_gif


@patch("src.utils.visualization.plt.show")
@patch("src.utils.visualization.plt.imshow")
def test_display_image(mock_imshow, mock_show):
    """Test image display with mocked matplotlib."""
    img = Image.new("RGB", (100, 100), color="red")

    display_image(img, title="Test")

    # Verify matplotlib functions were called
    mock_imshow.assert_called_once()
    mock_show.assert_called_once()

    # Verify the image was passed to imshow
    call_args = mock_imshow.call_args[0]
    assert call_args[0] == img


@patch("src.utils.visualization.plt.show")
@patch("src.utils.visualization.plt.imshow")
def test_display_sequence(mock_imshow, mock_show):
    """Test sequence display with mocked matplotlib."""
    images = [Image.new("RGB", (50, 50), color="blue") for _ in range(3)]

    display_sequence(images, title="Test", max_cols=2, sampling_rate=1)

    # Verify show was called once (for the whole grid)
    mock_show.assert_called_once()

    # Verify imshow was called for each image
    assert mock_imshow.call_count == len(images)


@patch("src.utils.visualization.display")
def test_display_gif(mock_display):
    """Test GIF display with mocked IPython display."""
    frames = [Image.new("RGB", (50, 50), color="green") for _ in range(2)]

    display_gif(frames, duration=100)

    # Verify IPython display was called
    mock_display.assert_called_once()


@patch("src.utils.visualization.plt.show")
def test_display_image_invalid_input(mock_show):
    """Test display_image handles invalid input gracefully."""
    # Should print error and return without calling show
    display_image("not an image", title="Test")

    mock_show.assert_not_called()


@patch("src.utils.visualization.plt.show")
def test_display_sequence_empty_list(mock_show):
    """Test display_sequence handles empty list gracefully."""
    display_sequence([], title="Test")

    # Should not call show for empty list
    mock_show.assert_not_called()


@patch("src.utils.visualization.plt.show")
@patch("src.utils.visualization.plt.imshow")
def test_display_sequence_sampling(mock_imshow, mock_show):
    """Test display_sequence with sampling rate."""
    images = [Image.new("RGB", (50, 50), color="blue") for _ in range(10)]

    # Display every 3rd image
    display_sequence(images, sampling_rate=3)

    # Should display: indices 0, 3, 6, 9 (last always included)
    expected_calls = 4
    assert mock_imshow.call_count == expected_calls
