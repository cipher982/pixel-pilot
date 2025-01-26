"""Tests for Docker-based GUI controller."""

import os

import pytest
from PIL import Image

from pixelpilot.gui_control_docker import DockerGUIController


@pytest.fixture(scope="session")
def docker_check():
    """Skip tests if not in Docker environment."""
    # These tests use X11-based GUI automation which only works in our Docker environment.
    # For local development, we use the native GUI controller implementation.
    if not os.path.exists("/.dockerenv"):
        pytest.skip("Skipping Docker GUI tests - these only run in Docker environment")


@pytest.fixture(scope="session")
def x11_display(docker_check):
    """Ensure X11 display is set."""
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":99"
    return os.environ["DISPLAY"]


@pytest.fixture
def controller(x11_display):
    """Create and cleanup controller for each test."""
    controller = None
    try:
        controller = DockerGUIController()
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
    """Test cleanup allows creating new controllers."""
    # Create and cleanup several controllers in sequence
    # If cleanup fails, subsequent controller creation will fail
    for _ in range(3):
        controller = DockerGUIController()
        assert controller.display is not None  # Verify connection works
        controller.cleanup()
