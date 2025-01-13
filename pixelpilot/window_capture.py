from typing import Any
from typing import Dict
from typing import Optional

import pyautogui
from PIL import Image

from pixelpilot.logger import setup_logger
from pixelpilot.utils import log_runtime

logger = setup_logger(__name__)


class WindowCapture:
    def __init__(self, debug: bool = False):
        pass

    @log_runtime
    def capture_fullscreen(self) -> Optional[Image.Image]:
        """Capture the entire screen and return as PIL Image."""

        # Capture full screen using pyautogui
        screenshot = pyautogui.screenshot()
        return screenshot

    def capture_window(
        self, window_info: Optional[Dict[str, Any]] = None, output_path: Optional[str] = None
    ) -> Optional[Image.Image]:
        """Capture the full screen and return as PIL Image."""
        screenshot = self.capture_fullscreen()

        if output_path and screenshot:
            screenshot.save(output_path)

        return screenshot
