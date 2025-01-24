"""Tests for X11-based GUI controller."""

import os

import pytest
from PIL import Image

from pixelpilot.gui_control_eval import EvalGUIController


def pytest_configure(config):
    """Skip tests if not in Docker environment."""
    if not os.path.exists("/.dockerenv"):
        pytest.skip("Skipping GUI tests outside Docker environment", allow_module_level=True)


@pytest.fixture(scope="session")
def x11_display():
    """Ensure X11 display is set."""
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":99"
    return os.environ["DISPLAY"]


@pytest.fixture
def controller(x11_display):
    """Create and cleanup controller for each test."""
    controller = None
    try:
        controller = EvalGUIController()
        yield controller
    finally:
        if controller is not None:
            controller.cleanup()


def test_x11_connection(controller):
    """Test basic X11 connection."""
    assert controller.display is not None
    assert controller.screen is not None
    assert controller.root is not None


def test_screen_capture(controller):
    """Test screen capture functionality."""
    image, result = controller.capture_screen()

    # Check operation result
    assert result.success
    assert "Screen captured successfully" in result.message
    assert "width" in result.details
    assert "height" in result.details

    # Check image
    assert image is not None
    assert isinstance(image, Image.Image)
    assert image.width == result.details["width"]
    assert image.height == result.details["height"]

    # Basic sanity check on image data
    assert image.mode == "RGB"
    # Don't check for non-zero content since Xvfb starts with a black screen
    assert image.size == (1024, 768), "Expected 1024x768 screen"

    # Verify we can access pixel data
    try:
        image.getpixel((0, 0))
    except Exception as e:
        pytest.fail(f"Could not access pixel data: {e}")


def test_cleanup(x11_display):
    """Test cleanup properly closes X11 connection."""
    controller = EvalGUIController()
    controller.cleanup()

    # Just verify we can create a new connection after cleanup
    new_controller = EvalGUIController()
    assert new_controller.display is not None
    new_controller.cleanup()
