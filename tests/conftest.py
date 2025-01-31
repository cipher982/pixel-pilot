"""Test configuration and fixtures."""

import logging
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def page_screenshot(tmp_path_factory):
    """Get screenshot and element coordinates from test page."""
    # Create a temporary directory for test artifacts
    tmp_dir = tmp_path_factory.mktemp("screenshots")
    screenshot_path = tmp_dir / "test_screenshot.png"

    with sync_playwright() as p:
        # Launch browser with fixed viewport for consistent coordinates
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})

        # Load the test page
        test_page = Path("tests/data/test_page.html")
        page.goto(f"file://{test_page.absolute()}")

        # Get button coordinates
        button = page.locator("#submit-btn")
        box = button.bounding_box()

        if box is None:
            raise ValueError("Failed to get button bounding box - button not found or not visible")

        logger.info("Button bounding box: %s", box)

        # Take screenshot
        page.screenshot(path=str(screenshot_path))

        # Clean up
        browser.close()

        return {
            "screenshot_path": screenshot_path,
            "button": {
                "x": int(box["x"]),
                "y": int(box["y"]),
                "width": int(box["width"]),
                "height": int(box["height"]),
            },
        }
