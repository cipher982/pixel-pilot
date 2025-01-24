"""Tests for native GUI controller."""

import pytest
from PIL import Image

from pixelpilot.gui_control_native import NativeGUIController


@pytest.fixture(scope="session", autouse=True)
def check_environment(is_host):
    """Skip these tests in Docker environment."""
    if not is_host:
        pytest.skip("Skipping host GUI tests in Docker environment")


@pytest.fixture
def controller():
    """Create controller for testing."""
    return NativeGUIController()


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
