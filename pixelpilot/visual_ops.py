import base64
import time
from io import BytesIO
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
from PIL import Image

from pixelpilot.action_utils import click_at_coordinates
from pixelpilot.action_utils import convert_relative_to_absolute
from pixelpilot.action_utils import scroll_action
from pixelpilot.logger import setup_logger
from pixelpilot.models import VisualAction
from pixelpilot.state_management import SharedState
from pixelpilot.utils import get_som_labeled_img
from pixelpilot.window_capture import WindowCapture

logger = setup_logger(__name__)


class VisualOperations:
    """Implements visual operations for the dual-path system."""

    def __init__(self, window_info: Optional[Dict[str, Any]]):
        self.window_info = window_info or {}  # Default to empty dict if None
        self.window_capture = WindowCapture()
        self.yolo_model = None  # Initialize YOLO model when needed

    def _ensure_context(self, state: SharedState) -> None:
        """Ensure the context dictionary exists in the state."""
        if "context" not in state:
            state["context"] = {}

    def execute_visual_action(self, state: SharedState) -> SharedState:
        """Execute a visual action based on LLM decision."""
        logger.info("Executing visual action...")

        # Get the action from state
        action_data = state["context"].get("next_action")
        if not action_data:
            logger.error("No action found in state")
            return state

        action = VisualAction(**action_data)

        try:
            # Update visual state first
            screenshot = self.window_capture.capture_window(self.window_info)
            if screenshot:
                state["screenshot"] = screenshot

            # Execute based on operation type
            if action.operation == "click":
                if action.coordinates:
                    abs_x, abs_y = convert_relative_to_absolute(
                        self.window_info, action.coordinates["x"], action.coordinates["y"]
                    )
                    click_at_coordinates(self.window_info, abs_x, abs_y, duration=0.5)
                    logger.info(f"Clicked at coordinates ({abs_x}, {abs_y})")

            elif action.operation == "scroll":
                scroll_action(self.window_info)
                logger.info("Performed scroll action")

            elif action.operation == "read":
                # Just capture the current state for analysis
                pass

            # Record action result
            state["context"]["last_action_result"] = {
                "success": True,
                "action": action.dict(),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Failed to execute visual action: {e}")
            state["context"]["last_action_result"] = {
                "success": False,
                "error": str(e),
                "action": action.dict(),
                "timestamp": time.time(),
            }

        return state

    def analyze_visual_result(self, state: SharedState) -> SharedState:
        """Analyze the result of the last visual action."""
        logger.info("Analyzing visual result...")

        # Get the last action and its result
        last_result = state["context"].get("last_action_result", {})
        if not last_result:
            logger.warning("No action result to analyze")
            return state

        # Process screenshot if available
        if state.get("screenshot"):
            screenshot_np = np.array(state["screenshot"])
            try:
                labeled_img, coordinates, _ = get_som_labeled_img(screenshot_np)

                # Store results in state
                if labeled_img is not None:
                    # Convert labeled image to base64 for storage
                    buffered = BytesIO()
                    Image.fromarray(np.uint8(labeled_img)).save(buffered, format="PNG")
                    state["labeled_img"] = base64.b64encode(buffered.getvalue()).decode()

                if coordinates:
                    state["label_coordinates"] = coordinates
            except Exception as e:
                logger.error(f"Failed to process screenshot: {e}")

        # Update task status based on action success
        if last_result.get("success"):
            state["task_status"] = "in_progress"
        else:
            state["task_status"] = "failed"
            state["context"]["error"] = last_result.get("error")

        return state
