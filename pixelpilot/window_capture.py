import os
import time
from typing import Any
from typing import Dict
from typing import Optional

import Quartz
import Quartz.CoreGraphics as CG
from PIL import Image

from pixelpilot.logger import setup_logger
from pixelpilot.utils import log_runtime

logger = setup_logger(__name__)


DEBUG_IMAGE = "./examples/quiz.png"


class WindowCapture:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.options = CG.kCGWindowListOptionOnScreenOnly
        if debug:
            self.debug_image = Image.open(DEBUG_IMAGE)

    @staticmethod
    def select_window_interactive(use_chrome: bool = False, use_firefox: bool = False) -> Optional[Dict[str, Any]]:
        """Interactive window selector that highlights windows as you move the mouse."""
        logger.info("Move your mouse over the window you want to capture and press Enter...")

        last_window = None  # Track the last window to prevent duplicate logging

        if use_chrome or use_firefox:
            """Quick hack to return browser window instead of interactive selection."""
            window_list = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID,
            )

            browser_name = "Google Chrome" if use_chrome else "Firefox"
            for window in window_list:
                if window.get("kCGWindowOwnerName") == browser_name:
                    logger.info(f"Found {browser_name} window")
                    return window

            logger.error(f"No {browser_name} window found")
            return None

        while True:
            try:
                # Check if Enter is pressed (non-blocking)
                import select
                import sys

                # Check if there's input ready (Enter pressed)
                if select.select([sys.stdin], [], [], 0.1)[0]:  # 0.1s timeout
                    if input() == "":
                        if last_window:
                            logger.info(f"Selected: {last_window.get('kCGWindowOwnerName', '')}")
                            return last_window
                        continue

                # Get the current mouse location
                mouse_loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))

                window_list = Quartz.CGWindowListCopyWindowInfo(
                    Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                    Quartz.kCGNullWindowID,
                )

                found_window = False
                for window in window_list:
                    bounds = window.get("kCGWindowBounds")
                    if bounds:
                        x = bounds["X"]
                        y = bounds["Y"]
                        width = bounds["Width"]
                        height = bounds["Height"]

                        if x <= mouse_loc.x <= x + width and y <= mouse_loc.y <= y + height:
                            found_window = True
                            if window != last_window:  # Only log if window changed
                                owner = window.get("kCGWindowOwnerName", "")
                                title = window.get("kCGWindowName", "")
                                display = f"{owner}"
                                if title:
                                    display += f" - {title}"
                                logger.info(f"Hovering: {display}")
                                last_window = window
                            break

                if not found_window and last_window is not None:
                    logger.info("No window under cursor")
                    last_window = None

            except KeyboardInterrupt:
                logger.info("Selection cancelled")
                return None

            time.sleep(0.1)  # Small delay to prevent high CPU usage

    @log_runtime
    def capture_window(self, window_info: Dict[str, Any], output_path: Optional[str] = None) -> Optional[Image.Image]:
        """Capture the specified window and return as PIL Image.

        Args:
            window_info: Window information dictionary
            output_path: Optional path to save image. If None, returns image without saving
        """
        if self.debug:
            return self.debug_image.copy()

        if not window_info:
            return None

        window_id = window_info["kCGWindowNumber"]

        # Create CGImage directly
        image_ref = CG.CGWindowListCreateImage(
            CG.CGRectNull,
            CG.kCGWindowListOptionIncludingWindow,
            window_id,
            CG.kCGWindowImageBoundsIgnoreFraming | CG.kCGWindowImageShouldBeOpaque,
        )

        # Convert to bitmap context to ensure proper pixel format
        width = CG.CGImageGetWidth(image_ref)
        height = CG.CGImageGetHeight(image_ref)
        context = CG.CGBitmapContextCreate(
            None,
            width,
            height,
            8,  # bits per component
            width * 4,  # bytes per row
            CG.CGColorSpaceCreateDeviceRGB(),
            CG.kCGImageAlphaPremultipliedLast,
        )

        # Draw the image in the bitmap context
        CG.CGContextDrawImage(context, CG.CGRectMake(0, 0, width, height), image_ref)

        # Get the image from the context
        image_ref = CG.CGBitmapContextCreateImage(context)
        pixel_data = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image_ref))

        # Create PIL Image from raw data
        image = Image.frombytes("RGBA", (width, height), pixel_data)

        if output_path:
            image.save(output_path)

        return image


if __name__ == "__main__":
    capture = WindowCapture()
    window_info = capture.select_window_interactive()
    if window_info:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.expanduser(f"./window_capture_{timestamp}.png")
        capture.capture_window(window_info, output_path)
    else:
        logger.error("No window selected")
