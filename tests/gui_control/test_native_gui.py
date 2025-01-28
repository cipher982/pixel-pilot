"""Tests for native system controller."""

import pytest
from PIL import Image

from pixelpilot.system_control_native import NativeSystemController


@pytest.fixture(scope="session", autouse=True)
def check_environment(is_host):
    """Skip these tests in Docker environment."""
    if not is_host:
        pytest.skip("Skipping host GUI tests in Docker environment")


@pytest.fixture
def controller():
    """Create controller for testing."""
    controller = NativeSystemController()
    try:
        result = controller.setup()
        assert result.success, f"Controller initialization failed: {result.message}"
        yield controller
    finally:
        controller.cleanup()


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
    # PyAutoGUI uses RGBA mode
    assert image.mode in ["RGB", "RGBA"]
    # Verify we can access pixel data
    try:
        image.getpixel((0, 0))
    except Exception as e:
        pytest.fail(f"Could not access pixel data: {e}")


@pytest.mark.skip(reason="Skip potentially disruptive UI tests by default")
def test_click(controller):
    """Test mouse click (just verify it doesn't error)."""
    # Click in a safe area (top-left corner)
    result = controller.click(x=0, y=0)
    assert result.success
    assert "Clicked at (0, 0)" in result.message


@pytest.mark.skip(reason="Skip potentially disruptive UI tests by default")
def test_type_text(controller):
    """Test typing (just verify it doesn't error)."""
    result = controller.type_text("test")
    assert result.success
    assert "Typed text: test" in result.message


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
