"""Visual operations module."""

from typing import Any
from typing import Dict
from typing import Optional

import pyautogui

from pixelpilot.logger import setup_logger
from pixelpilot.models import Action
from pixelpilot.state_management import SharedState

logger = setup_logger(__name__)


class VisualOperations:
    """Handles visual operations like clicking and scrolling."""

    def __init__(self, window_info: Optional[Dict[str, Any]] = None):
        """Initialize visual operations."""
        self.window_info = window_info or {}

    def execute_visual_action(self, state: SharedState) -> SharedState:
        """Execute a visual action."""
        action = Action(**state["context"]["next_action"])
        if action.type != "visual":
            raise ValueError("Expected visual action")

        args = action.args or {}
        operation = args.get("operation")
        coordinates = args.get("coordinates", {})

        try:
            if operation == "click":
                x, y = coordinates.get("x", 0), coordinates.get("y", 0)
                pyautogui.click(x=x, y=y)
                state["context"]["last_action_result"] = {"success": True, "output": f"Clicked at ({x}, {y})"}
            elif operation == "scroll":
                amount = coordinates.get("amount", 0)
                pyautogui.scroll(amount)
                state["context"]["last_action_result"] = {"success": True, "output": f"Scrolled {amount} units"}
            else:
                raise ValueError(f"Unknown visual operation: {operation}")

        except Exception as e:
            logger.error(f"Visual operation failed: {e}")
            state["context"]["last_action_result"] = {"success": False, "output": None, "error": str(e)}

        return state

    def analyze_visual_result(self, state: SharedState) -> SharedState:
        """Analyze the result of a visual operation."""
        result = state["context"].get("last_action_result", {})

        if not result.get("success", False):
            logger.error(f"Visual operation failed: {result.get('error')}")
            return state

        # For now, visual operations don't determine task completion
        # This could be enhanced with screenshot analysis or other checks
        return state
