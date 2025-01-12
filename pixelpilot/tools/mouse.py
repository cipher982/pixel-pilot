"""Mouse interaction tool."""

import subprocess
import time
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Type

import pyautogui
from langchain.tools import BaseTool
from pydantic import BaseModel
from pydantic import Field

from pixelpilot.logger import setup_logger

logger = setup_logger(__name__)


class MouseInput(BaseModel):
    """Schema for mouse input."""

    action: Literal["click", "scroll"] = Field(description="The action to perform")
    element_id: Optional[str] = Field(None, description="Element ID from vision model")
    coordinates: Optional[Tuple[float, float]] = Field(None, description="Relative coordinates (0-1)")
    scroll_amount: Optional[int] = Field(None, description="Amount to scroll")
    window_info: Dict[str, Any] = Field(description="Window information for targeting")
    labeled_coordinates: Optional[Dict[str, Any]] = Field(None, description="Vision model output")


class MouseTool(BaseTool):
    """Tool for mouse control."""

    name: str = "mouse"
    description: str = "Control mouse for clicking and scrolling"
    args_schema: Type[BaseModel] = MouseInput

    def __init__(self):
        super().__init__()
        self.last_window = None
        self.last_coordinates = None
        pyautogui.FAILSAFE = True

    def _convert_relative_to_absolute(self, window_info: Dict[str, Any], rel_x: float, rel_y: float) -> Tuple[int, int]:
        """Convert relative coordinates to absolute screen coordinates."""
        window_bounds = window_info["kCGWindowBounds"]
        window_x = window_bounds["X"]
        window_y = window_bounds["Y"]
        window_width = window_bounds["Width"]
        window_height = window_bounds["Height"]

        abs_x = window_x + (rel_x * window_width)
        abs_y = window_y + (rel_y * window_height)

        return int(abs_x), int(abs_y)

    def _focus_window(self, window_info: Dict[str, Any]) -> bool:
        """Focus the target window."""
        try:
            script = f"""
                tell application "System Events"
                    tell process "{window_info['kCGWindowOwnerName']}"
                        set frontmost to true
                    end tell
                end tell
            """
            subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
            time.sleep(0.2)  # Brief pause to let window focus take effect
            return True
        except Exception as e:
            logger.error(f"Failed to focus window: {e}")
            return False

    def _click(
        self,
        window_info: Dict[str, Any],
        coordinates: Optional[Tuple[float, float]] = None,
        element_id: Optional[str] = None,
        labeled_coordinates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a click action."""
        try:
            # Focus window first
            if not self._focus_window(window_info):
                return {"success": False, "error": "Failed to focus window"}

            # Get coordinates either from direct input or element_id
            if coordinates:
                rel_x, rel_y = coordinates
            elif element_id and labeled_coordinates:
                coords = labeled_coordinates.get(element_id)
                if coords is None:
                    return {"success": False, "error": f"Element id {element_id} not found in coordinates"}
                # Use center of the bounding box
                rel_x = coords[0] + (coords[2] / 2)
                rel_y = coords[1] + (coords[3] / 2)
            else:
                return {"success": False, "error": "No valid coordinates or element_id provided"}

            # Convert to absolute coordinates
            abs_x, abs_y = self._convert_relative_to_absolute(window_info, rel_x, rel_y)

            # Perform click
            pyautogui.moveTo(abs_x, abs_y, duration=0.3)
            pyautogui.click()
            time.sleep(0.5)  # Wait for click to register

            self.last_coordinates = (abs_x, abs_y)
            logger.info(f"Clicked at ({abs_x}, {abs_y})")

            return {"success": True, "coordinates": (abs_x, abs_y)}

        except Exception as e:
            logger.error(f"Click failed: {e}")
            return {"success": False, "error": str(e)}

    def _scroll(self, window_info: Dict[str, Any], scroll_amount: Optional[int] = None) -> Dict[str, Any]:
        """Execute a scroll action."""
        try:
            # Focus window first
            if not self._focus_window(window_info):
                return {"success": False, "error": "Failed to focus window"}

            # Default scroll amount if not specified
            amount = scroll_amount if scroll_amount is not None else 300

            # Send scroll command
            pyautogui.scroll(amount)  # Positive scrolls up, negative scrolls down
            time.sleep(0.5)  # Wait for scroll to complete

            logger.info(f"Scrolled by {amount} units")
            return {"success": True, "scroll_amount": amount}

        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return {"success": False, "error": str(e)}

    def _run(
        self,
        action: str,
        window_info: Dict[str, Any],
        element_id: Optional[str] = None,
        coordinates: Optional[Tuple[float, float]] = None,
        scroll_amount: Optional[int] = None,
        labeled_coordinates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a mouse action.

        Args:
            action: Type of action (click/scroll)
            window_info: Window information for targeting
            element_id: Optional element ID from vision model
            coordinates: Optional relative coordinates (0-1)
            scroll_amount: Optional scroll amount
            labeled_coordinates: Optional vision model output

        Returns:
            Dict containing action results
        """
        # Validate window info
        if not window_info:
            return {"success": False, "error": "No window info provided"}

        # Store window info
        self.last_window = window_info

        # Execute requested action
        if action == "click":
            return self._click(
                window_info=window_info,
                coordinates=coordinates,
                element_id=element_id,
                labeled_coordinates=labeled_coordinates,
            )
        elif action == "scroll":
            return self._scroll(window_info=window_info, scroll_amount=scroll_amount)
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
