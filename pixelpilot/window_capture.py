"""Window capture module for Docker/X11 environment."""

import os
from typing import Any
from typing import Dict
from typing import Optional

import Xlib.display
import Xlib.X
from PIL import Image
from Xlib import error as Xerror

from pixelpilot.logger import setup_logger
from pixelpilot.utils import log_runtime

logger = setup_logger(__name__)


class WindowCapture:
    """Handles window capture in Docker/X11 environment."""

    def __init__(self):
        """Initialize with X11 display connection."""
        self.display_str = os.environ.get("DISPLAY", ":0")
        try:
            self.display = Xlib.display.Display(self.display_str)
            self.screen = self.display.screen()
            logger.info(f"Connected to X display {self.display_str}")
            logger.info(f"Screen dimensions: {self.screen.width_in_pixels}x{self.screen.height_in_pixels}")
        except Xerror.DisplayConnectionError as e:
            logger.error(f"Failed to connect to X display {self.display_str}: {e}")
            raise

    @log_runtime
    def capture_fullscreen(self) -> Optional[Image.Image]:
        """Capture the entire X11 screen and return as PIL Image."""
        try:
            root = self.display.screen().root
            geometry = root.get_geometry()

            # Get raw pixels from X11
            raw = root.get_image(
                0,
                0,  # x, y
                geometry.width,
                geometry.height,  # width, height
                Xlib.X.ZPixmap,
                0xFFFFFFFF,  # format, plane_mask
            )

            # Convert to PIL Image
            image = Image.frombytes(
                "RGB",
                (geometry.width, geometry.height),
                raw.data,
                "raw",
                "BGRX",  # X11 uses BGRX format
            )

            return image

        except Xerror.XError as e:
            logger.error(f"X11 error during capture: {e}")
            return None

    def capture_window(
        self, window_info: Optional[Dict[str, Any]] = None, output_path: Optional[str] = None
    ) -> Optional[Image.Image]:
        """Capture a specific window or full screen."""
        # TODO: Implement window-specific capture using window_info
        # For now, just capture full screen
        screenshot = self.capture_fullscreen()

        if output_path and screenshot:
            screenshot.save(output_path)

        return screenshot

    def __del__(self):
        """Clean up X11 connection."""
        if hasattr(self, "display"):
            self.display.close()
