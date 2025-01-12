import os
import time
from typing import Any
from typing import Dict
from typing import Optional

import pyautogui
from PIL import Image

from pixelpilot.logger import setup_logger
from pixelpilot.utils import log_runtime

logger = setup_logger(__name__)


DEBUG_IMAGE = "./examples/quiz.png"


class WindowCapture:
    def __init__(self, debug: bool = False):
        self.debug = debug
        if debug:
            self.debug_image = Image.open(DEBUG_IMAGE)

    @log_runtime
    def capture_fullscreen(self) -> Optional[Image.Image]:
        """Capture the entire screen and return as PIL Image."""
        if self.debug:
            return self.debug_image.copy()

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


if __name__ == "__main__":
    capture = WindowCapture()
    window_info = capture.select_window_interactive()
    if window_info:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.expanduser(f"./window_capture_{timestamp}.png")
        capture.capture_window(window_info, output_path)
    else:
        logger.error("No window selected")
