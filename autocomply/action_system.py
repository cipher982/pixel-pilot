from typing import Any
from typing import Optional
from typing import Tuple

import numpy as np
import pyautogui
import pygetwindow as gw

from autocomply.logger import setup_logger

logger = setup_logger(__name__)


class ActionSystem:
    def __init__(self):
        # Ensure PyAutoGUI fails safe
        pyautogui.FAILSAFE = True

    def click_at_position(self, window: Any, relative_pos: Tuple[int, int]) -> None:
        """Click at a position relative to the window."""
        x = window.left + relative_pos[0]
        y = window.top + relative_pos[1]
        logger.debug(f"Clicking at position: ({x}, {y})")
        pyautogui.moveTo(x, y)
        pyautogui.click()

    def decide_action(
        self, transcribed_text: Optional[str], screenshot: Optional[np.ndarray]
    ) -> Optional[Tuple[str, Tuple[int, int]]]:
        """
        Simple decision system for MVP.
        Returns action type and coordinates if action needed.
        """
        if not transcribed_text:
            return None

        # MVP: Just look for specific keywords
        transcribed_text = transcribed_text.lower()
        if "next" in transcribed_text or "continue" in transcribed_text:
            logger.debug("Detected 'next/continue' command")
            # For MVP, assume next button is always at this position
            return ("click", (100, 200))

        return None


if __name__ == "__main__":
    # Simple test
    action_sys = ActionSystem()
    try:
        test_window = gw.getWindowsWithTitle("Google Chrome")[0]
        logger.info("Testing decision system...")
        action = action_sys.decide_action("Please click next", None)
        if action:
            logger.info(f"Decided to perform action: {action}")
            if action[0] == "click":
                action_sys.click_at_position(test_window, action[1])
    except Exception as e:
        logger.error(f"Test failed: {e}")
