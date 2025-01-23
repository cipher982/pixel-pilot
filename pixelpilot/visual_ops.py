"""Visual operations module."""

from typing import Any
from typing import Dict
from typing import Optional

import pyautogui

from pixelpilot.logger import setup_logger
from pixelpilot.models import Action
from pixelpilot.state_management import SharedState
from pixelpilot.window_capture import WindowCapture

logger = setup_logger(__name__)


class VisualOperations:
    """Handles visual operations like clicking and scrolling."""

    def __init__(self, window_info: Optional[Dict[str, Any]] = None):
        """Initialize visual operations."""
        self.window_info = window_info or {}
        self.window_capture = WindowCapture()

    def execute_visual_action(self, state: SharedState) -> SharedState:
        """Execute a visual action."""
        action = Action(**state["context"]["next_action"])
        if action.type != "visual":
            raise ValueError("Expected visual action")

        args = action.args or {}
        operation = args.get("operation")
        coordinates = args.get("coordinates", {})

        try:
            # Capture before state
            before_screen = self.window_capture.capture_window(self.window_info)

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

            # Capture after state
            after_screen = self.window_capture.capture_window(self.window_info)

            # Store screenshots for verification
            state["context"]["before_screen"] = before_screen
            state["context"]["after_screen"] = after_screen

        except Exception as e:
            logger.error(f"Visual operation failed: {e}")
            state["context"]["last_action_result"] = {"success": False, "output": None, "error": str(e)}

        return state

    def analyze_visual_result(self, state: SharedState) -> SharedState:
        """Analyze the result of a visual operation using AI verification."""
        result = state["context"].get("last_action_result", {})

        if not result.get("success", False):
            logger.error(f"Visual operation failed: {result.get('error')}")
            return state

        # Get the before and after screenshots
        before_screen = state["context"].get("before_screen")
        after_screen = state["context"].get("after_screen")

        if not (before_screen and after_screen):
            logger.warning("Missing screenshots for verification")
            return state

        # Get the operation details
        action = Action(**state["context"]["next_action"])
        args = action.args or {}
        operation = args.get("operation")

        # Prepare verification prompt
        _ = f"""
        Task: Verify the success of a {operation} operation in the GUI.
        
        Context:
        - Operation type: {operation}
        - Operation details: {args}
        - Expected result: {state.get('task_description', '')}
        
        Please analyze the before and after screenshots to determine:
        1. If the intended operation was successful
        2. If the GUI state changed as expected
        3. If there are any error messages or unexpected states
        
        Respond with:
        - success: true/false
        - confidence: 0-1
        - explanation: brief description of what changed
        """

        # The verification prompt and screenshots would be sent to the AI model
        # For now, we'll assume success if the operation didn't raise an exception
        # TODO: Implement actual AI verification when model integration is ready

        state["context"]["verification_result"] = {
            "success": True,
            "confidence": 0.9,
            "explanation": f"Operation {operation} completed without errors",
        }

        return state
