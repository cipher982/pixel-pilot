"""X11-based GUI controller for eval environment."""

import os
from typing import Any
from typing import Optional
from typing import Tuple
from typing import cast

import Xlib.display
import Xlib.ext.xtest
import Xlib.X
import Xlib.XK
from PIL import Image
from Xlib import error as Xerror
from Xlib.display import Display

from pixelpilot.gui_control import GUIController
from pixelpilot.gui_control import OperationResult
from pixelpilot.logger import setup_logger

logger = setup_logger(__name__)


class EvalGUIController(GUIController):
    """GUI Controller implementation for X11/Docker environment."""

    def __init__(self):
        """Initialize controller."""
        self.display: Optional[Display] = None
        self.screen = None
        self.root: Optional[Any] = None  # Xlib.Window type is complex, use Any
        # Initialize right away
        self.initialize()

    def initialize(self) -> OperationResult:
        """Initialize X11 connection."""
        try:
            self.display = Xlib.display.Display(os.environ.get("DISPLAY", ":0"))
            self.screen = self.display.screen()
            self.root = self.screen.root

            # Test connection with basic operation
            size = self.get_screen_size()
            logger.info(f"Connected to X11 display, screen size: {size[0]}x{size[1]}")

            return OperationResult(success=True, message="X11 connection established", details={"screen_size": size})
        except Xerror.DisplayConnectionError as e:
            logger.error(f"Failed to connect to X11 display: {e}")
            return OperationResult(success=False, message=f"X11 connection failed: {str(e)}")

    def capture_screen(self) -> Tuple[Optional[Image.Image], OperationResult]:
        """Capture screen via X11."""
        if not all([self.display, self.screen, self.root]):
            return None, OperationResult(success=False, message="X11 not initialized")

        try:
            # After the check above, we know these aren't None
            display = cast(Display, self.display)
            root = self.root  # Already checked for None

            geometry = root.get_geometry()

            # Get raw pixels
            raw = root.get_image(0, 0, geometry.width, geometry.height, Xlib.X.ZPixmap, 0xFFFFFFFF)

            # Convert to PIL Image
            image = Image.frombytes(
                "RGB",
                (geometry.width, geometry.height),
                raw.data,
                "raw",
                "BGRX",  # X11 uses BGRX format
            )

            # Ensure display is synced after operation
            display.sync()

            return image, OperationResult(success=True, message="Screen captured successfully")

        except Xerror.XError as e:
            logger.error(f"Screen capture failed: {e}")
            return None, OperationResult(success=False, message=f"Screen capture failed: {str(e)}")

    def click(self, x: int, y: int) -> OperationResult:
        """Perform mouse click via X11."""
        if not all([self.display, self.screen, self.root]):
            return OperationResult(success=False, message="X11 not initialized")

        try:
            # After the check above, we know these aren't None
            display = cast(Display, self.display)
            root = self.root  # Already checked for None

            # Move pointer
            root.warp_pointer(x, y)
            display.sync()

            # Click events
            button = 1  # Left click

            # Press
            Xlib.ext.xtest.fake_input(display, Xlib.X.ButtonPress, button)
            # Release
            Xlib.ext.xtest.fake_input(display, Xlib.X.ButtonRelease, button)

            display.sync()

            return OperationResult(success=True, message=f"Clicked at ({x}, {y})")

        except Xerror.XError as e:
            logger.error(f"Click operation failed: {e}")
            return OperationResult(success=False, message=f"Click failed: {str(e)}")

    def scroll(self, amount: int) -> OperationResult:
        """Scroll via X11."""
        if not all([self.display, self.screen, self.root]):
            return OperationResult(success=False, message="X11 not initialized")

        try:
            # After the check above, we know this isn't None
            display = cast(Display, self.display)

            button = 4 if amount > 0 else 5  # 4 = up, 5 = down

            # Convert amount to number of scroll events
            events = abs(amount)

            for _ in range(events):
                Xlib.ext.xtest.fake_input(display, Xlib.X.ButtonPress, button)
                Xlib.ext.xtest.fake_input(display, Xlib.X.ButtonRelease, button)

            display.sync()

            return OperationResult(success=True, message=f"Scrolled {amount} units")

        except Xerror.XError as e:
            logger.error(f"Scroll operation failed: {e}")
            return OperationResult(success=False, message=f"Scroll failed: {str(e)}")

    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions from X11."""
        if not self.screen:
            raise RuntimeError("X11 not initialized")
        return (self.screen.width_in_pixels, self.screen.height_in_pixels)

    def cleanup(self) -> None:
        """Clean up X11 connection."""
        if self.display:
            self.display.close()
            self.display = None
            self.screen = None
            self.root = None
