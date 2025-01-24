"""Docker-based GUI controller using X11."""

import os
import re
from typing import Optional
from typing import Tuple

import Xlib.display
import Xlib.ext.xtest
import Xlib.X
from PIL import Image

from pixelpilot.gui_control import GUIController
from pixelpilot.gui_control import OperationResult
from pixelpilot.logger import setup_logger

logger = setup_logger(__name__)


def _normalize_display(display: str) -> str:
    """Convert XQuartz socket path to standard display format."""
    # Check if it's a Unix domain socket path (XQuartz style)
    if display.startswith("/"):
        # Extract display number from path
        match = re.search(r":(\d+)$", display)
        if match:
            return f":{match.group(1)}"
    return display


class DockerGUIController(GUIController):
    """Docker-based GUI controller using X11 display server."""

    def __init__(self):
        """Initialize X11 connection."""
        display_name = _normalize_display(os.environ.get("DISPLAY", ":0"))
        self.display = Xlib.display.Display(display_name)
        self.screen = self.display.screen()
        self.root = self.screen.root

        # Test connection
        try:
            self.root.get_geometry()
            logger.info("Connected to X11 display")
        except Exception as e:
            logger.error(f"Failed to connect to X11: {e}")
            raise

    def capture_screen(self) -> Tuple[Optional[Image.Image], OperationResult]:
        """Capture current screen state."""
        try:
            # Get window geometry
            geom = self.root.get_geometry()

            # Get raw image data
            raw = self.root.get_image(
                0,
                0,  # x, y
                geom.width,
                geom.height,
                Xlib.X.ZPixmap,
                0xFFFFFFFF,  # plane mask (all planes)
            )

            # Convert raw data to bytes if it's not already
            raw_data = raw.data
            if isinstance(raw_data, str):
                raw_data = raw_data.encode("latin-1")

            # Convert to PIL Image
            image = Image.frombytes(
                "RGB",
                (geom.width, geom.height),
                raw_data,
                "raw",
                "BGRX",  # X11 uses BGRX format
            )

            return image, OperationResult(
                success=True,
                message="Screen captured successfully",
                details={"width": geom.width, "height": geom.height},
            )

        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None, OperationResult(success=False, message=f"Screen capture failed: {str(e)}")

    def click(self, x: int, y: int) -> OperationResult:
        """Click at given coordinates."""
        # TODO: Implement with XTEST
        return OperationResult(success=False, message="Click not yet implemented")

    def type_text(self, text: str) -> OperationResult:
        """Type text at current cursor position."""
        # TODO: Implement with XTEST
        return OperationResult(success=False, message="Type text not yet implemented")

    def cleanup(self) -> None:
        """Clean up X11 resources."""
        if hasattr(self, "display"):
            self.display.close()
