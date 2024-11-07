import os
import time
from typing import Any
from typing import Dict
from typing import Optional

import Quartz
import Quartz.CoreGraphics as CG
from PIL import Image


class WindowCapture:
    def __init__(self):
        self.options = CG.kCGWindowListOptionOnScreenOnly

    @staticmethod
    def select_window_interactive() -> Optional[Dict[str, Any]]:
        """Interactive window selector that highlights windows as you move the mouse."""
        print("\nMove your mouse over the window you want to capture and press Enter...")

        while True:
            # Get the current mouse location
            mouse_loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))

            # Get the window under the mouse
            window_list = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID,
            )

            for window in window_list:
                bounds = window.get("kCGWindowBounds")
                if bounds:
                    x = bounds["X"]
                    y = bounds["Y"]
                    width = bounds["Width"]
                    height = bounds["Height"]

                    # Check if mouse is inside this window
                    if x <= mouse_loc.x <= x + width and y <= mouse_loc.y <= y + height:
                        owner = window.get("kCGWindowOwnerName", "")
                        title = window.get("kCGWindowName", "")
                        display = f"{owner}"
                        if title:
                            display += f" - {title}"
                        print(f"\rHovering: {display}", end="", flush=True)

                        # Check if Enter is pressed
                        try:
                            if input() == "":
                                print(f"\nSelected: {display}")
                                return window
                        except KeyboardInterrupt:
                            print("\nCancelled")
                            return None

            time.sleep(0.1)  # Small delay to prevent high CPU usage

    def get_chrome_window(self) -> Optional[Dict[str, Any]]:
        """Retrieve information about the Chrome window."""
        return self.get_window_info(self.window_name)

    def get_window_info(self, window_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve information about the specified window."""
        window_list = CG.CGWindowListCopyWindowInfo(self.options, CG.kCGNullWindowID)
        for window in window_list:
            if window_name.lower() in window.get("kCGWindowName", "").lower():
                return window
        return None

    def capture_window(self, window_info: Dict[str, Any], output_path: Optional[str] = None) -> Optional[Image.Image]:
        """Capture the specified window and return as PIL Image or save to file if output_path provided."""
        if not window_info:
            return None

        bounds = window_info["kCGWindowBounds"]
        x, y, width, height = (int(bounds[key]) for key in ("X", "Y", "Width", "Height"))

        # Create a bitmap context
        context = CG.CGBitmapContextCreate(
            None, width, height, 8, width * 4, CG.CGColorSpaceCreateDeviceRGB(), CG.kCGImageAlphaPremultipliedFirst
        )

        # Capture the window content
        CG.CGWindowListCreateImage(
            (x, y, x + width, y + height),
            CG.kCGWindowListOptionIncludingWindow,
            window_info["kCGWindowNumber"],
            CG.kCGWindowImageDefault,
        )

        # Create an image from the context
        image_ref = CG.CGBitmapContextCreateImage(context)
        if not image_ref:
            return None

        image = Image.frombytes(
            "RGBA",
            (width, height),
            CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image_ref)),
            "raw",
            "BGRA",
            0,
            1,
        )

        if output_path:
            image.save(output_path)
            print(f"Window captured and saved to {output_path}")

        return image


if __name__ == "__main__":
    # Example usage
    capture = WindowCapture()
    window_info = capture.get_chrome_window()
    if window_info:
        output_path = os.path.expanduser("~/Desktop/window_capture.png")
        capture.capture_window(window_info, output_path)
    else:
        print("No Chrome window found.")
