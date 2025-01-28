"""Tests for Docker-based system controller."""

import os

import pytest
from PIL import Image

from pixelpilot.system_control_docker import DockerSystemController


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


def test_click(controller):
    """Test mouse click."""
    result = controller.click(x=0, y=0)
    assert result.success
    assert "Clicked at (0, 0)" in result.message


def test_type_text(controller):
    """Test typing (should return not implemented)."""
    result = controller.type_text("test")
    assert not result.success
    assert "not yet implemented" in result.message


def test_run_command(controller):
    """Test running a simple command."""
    result = controller.run_command("echo test")
    assert result.success
    assert "test" in result.message


def test_file_operations(controller, tmp_path):
    """Test file operations."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_content = "test content"
    test_file.write_text(test_content)

    # Test file exists
    assert controller.file_exists(str(test_file))

    # Test read file
    assert controller.read_file(str(test_file)) == test_content

    # Test file size
    assert controller.get_file_size(str(test_file)) == len(test_content)


def test_cleanup(x11_display):
    """Test cleanup allows creating new controllers."""
    # Create and cleanup several controllers in sequence
    # If cleanup fails, subsequent controller creation will fail
    for _ in range(3):
        controller = DockerSystemController()
        result = controller.setup()
        assert result.success, f"Controller initialization failed: {result.message}"
        assert controller.display is not None  # Verify connection works
        controller.cleanup()
