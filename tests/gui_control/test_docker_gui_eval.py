"""Tests for Docker system controller in evaluation environment."""

import os

import pytest
from PIL import Image

from pixelpilot.system_control_docker import DockerSystemController


@pytest.fixture(scope="session", autouse=True)
def check_environment(is_docker):
    """Skip these tests outside Docker environment."""
    if not is_docker:
        pytest.skip("Skipping Docker GUI tests outside Docker environment")


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
        controller = DockerSystemController()
        result = controller.setup()
        assert result.success, f"Controller initialization failed: {result.message}"
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
    assert image.size == (1280, 720), "Expected 1280x720 screen"

    # Verify we can access pixel data
    try:
        image.getpixel((0, 0))
    except Exception as e:
        pytest.fail(f"Could not access pixel data: {e}")


def test_cleanup(x11_display):
    """Test cleanup properly closes X11 connection."""
    controller = DockerSystemController()
    result = controller.setup()
    assert result.success, f"Controller initialization failed: {result.message}"
    controller.cleanup()

    # Just verify we can create a new connection after cleanup
    new_controller = DockerSystemController()
    result = new_controller.setup()
    assert result.success, f"Controller initialization failed: {result.message}"
    new_controller.cleanup()
