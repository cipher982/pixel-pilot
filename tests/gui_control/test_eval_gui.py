"""Tests for X11-based GUI controller."""

import pytest
from PIL import Image


def test_screen_capture(x11_server, gui_controller):
    """Test basic screen capture functionality."""
    # Capture screen
    image, result = gui_controller.capture_screen()

    # Verify operation succeeded
    assert result.success, f"Screen capture failed: {result.message}"
    assert isinstance(image, Image.Image), "Expected PIL Image"

    # Verify image dimensions match reported screen size
    width, height = gui_controller.get_screen_size()
    assert image.width == width, f"Image width {image.width} != screen width {width}"
    assert image.height == height, f"Image height {image.height} != screen height {height}"


def test_mouse_click(x11_server, gui_controller):
    """Test mouse click operation."""
    # Try clicking in the middle of the screen
    width, height = gui_controller.get_screen_size()
    x, y = width // 2, height // 2

    result = gui_controller.click(x, y)
    assert result.success, f"Click operation failed: {result.message}"

    # Verify result contains coordinates
    assert result.message.startswith("Clicked at"), "Expected click confirmation message"
    assert str(x) in result.message and str(y) in result.message, "Click coordinates not in message"


def test_scroll(x11_server, gui_controller):
    """Test scroll operation."""
    # Test scroll up
    result = gui_controller.scroll(5)
    assert result.success, f"Scroll up failed: {result.message}"
    assert "5" in result.message, "Scroll amount not in message"

    # Test scroll down
    result = gui_controller.scroll(-3)
    assert result.success, f"Scroll down failed: {result.message}"
    assert "-3" in result.message, "Scroll amount not in message"


def test_screen_size(x11_server, gui_controller):
    """Test screen size retrieval."""
    width, height = gui_controller.get_screen_size()

    # Our test X server is 1024x768
    assert width == 1024, f"Expected width 1024, got {width}"
    assert height == 768, f"Expected height 768, got {height}"


def test_cleanup(x11_server, gui_controller):
    """Test cleanup behavior."""
    # Capture initial state
    result = gui_controller.click(0, 0)
    assert result.success, "Initial operation failed"

    # Cleanup
    gui_controller.cleanup()

    # Verify operations fail after cleanup
    with pytest.raises(RuntimeError, match="X11 not initialized"):
        gui_controller.get_screen_size()


def test_error_handling(x11_server, gui_controller):
    """Test error handling for invalid operations."""
    # Test invalid coordinates
    result = gui_controller.click(-1, -1)
    assert not result.success, "Expected failure for invalid coordinates"

    # Test massive scroll
    result = gui_controller.scroll(1000000)
    assert not result.success, "Expected failure for unreasonable scroll"
