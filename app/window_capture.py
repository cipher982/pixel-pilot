import os
from typing import Any
from typing import Dict
from typing import Optional

import Quartz.CoreGraphics as CG
from PIL import Image


class WindowCapture:
    def __init__(self):
        self.options = CG.kCGWindowListOptionOnScreenOnly

    def get_chrome_window(self) -> Optional[Dict[str, Any]]:
        """Retrieve information about the Chrome window."""
        return self.get_window_info("Google Chrome")

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
