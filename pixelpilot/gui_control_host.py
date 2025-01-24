"""Host GUI controller using PyAutoGUI."""

from typing import Optional
from typing import Tuple

import pyautogui
from PIL import Image

from pixelpilot.gui_control import GUIController
from pixelpilot.gui_control import OperationResult
from pixelpilot.logger import setup_logger

logger = setup_logger(__name__)


class HostGUIController(GUIController):
    """Native GUI control using PyAutoGUI."""

    def __init__(self):
        """Initialize controller."""
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = 0.1  # Small delay between actions

    def capture_screen(self) -> Tuple[Optional[Image.Image], OperationResult]:
        """Capture current screen state."""
        try:
            screenshot = pyautogui.screenshot()
            return screenshot, OperationResult(
                success=True,
                message="Screen captured successfully",
                details={"width": screenshot.width, "height": screenshot.height},
            )
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None, OperationResult(
                success=False,
                message=f"Screen capture failed: {str(e)}",
            )

    def click(self, x: int, y: int) -> OperationResult:
        """Click at given coordinates."""
        try:
            pyautogui.click(x=x, y=y)
            return OperationResult(
                success=True,
                message=f"Clicked at ({x}, {y})",
            )
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return OperationResult(
                success=False,
                message=f"Click failed: {str(e)}",
            )

    def type_text(self, text: str) -> OperationResult:
        """Type text at current cursor position."""
        try:
            pyautogui.typewrite(text)
            return OperationResult(
                success=True,
                message=f"Typed text: {text}",
            )
        except Exception as e:
            logger.error(f"Type text failed: {e}")
            return OperationResult(
                success=False,
                message=f"Type text failed: {str(e)}",
            )

    def cleanup(self) -> None:
        """Clean up resources."""
        # PyAutoGUI doesn't need cleanup
        pass
