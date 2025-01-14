import subprocess
import time
from typing import Any
from typing import Dict
from typing import Optional

from pixelpilot.logger import setup_logger

logger = setup_logger(__name__)


def back_action(window_info: Optional[Dict[str, Any]]) -> Any:
    """Navigate back in the web browser history."""
    logger.info("BACK action received")

    # Skip if no window info
    if not window_info:
        logger.warning("No window info provided, skipping back action")
        return

    # Send Cmd+Left Arrow to go back
    script = f"""
        tell application "System Events"
            tell process "{window_info['kCGWindowOwnerName']}"
                set frontmost to true
                key code 123 using command down  # Left arrow key
            end tell
        end tell
    """

    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
    logger.info(f"Back navigation result: {result.stdout}")
    time.sleep(0.5)  # Wait for navigation to complete
    logger.info("Navigated back")


def scroll_action(window_info: Optional[Dict[str, Any]]) -> Any:
    import pyautogui

    # Focus window first if window info is provided
    if window_info:
        # Simple keystroke command for each key
        key_commands = "\n".join([f'keystroke "{key}"' for key in ["down"]])

        script = f"""
            tell application "System Events"
                tell process "{window_info['kCGWindowOwnerName']}"
                    set frontmost to true
                    {key_commands}
                end tell
            end tell
        """

        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
        logger.info(f"AppleScript result: {result.stdout}")
        time.sleep(0.1)

    amount = 300
    pyautogui.scroll(amount)  # Negative values scroll down
    time.sleep(0.5)  # Wait for scroll to complete
    logger.info(f"Scrolled by {amount} units")


# Helper methods
def convert_relative_to_absolute(window_info: Optional[Dict[str, Any]], rel_x: float, rel_y: float) -> tuple[int, int]:
    """Convert relative coordinates to absolute screen coordinates."""
    if not window_info:
        # For full screen, use screen dimensions
        import pyautogui

        screen_width, screen_height = pyautogui.size()
        abs_x = int(rel_x * screen_width)
        abs_y = int(rel_y * screen_height)
        return abs_x, abs_y

    window_bounds = window_info["kCGWindowBounds"]
    window_x = window_bounds["X"]
    window_y = window_bounds["Y"]
    window_width = window_bounds["Width"]
    window_height = window_bounds["Height"]

    abs_x = window_x + (rel_x * window_width)
    abs_y = window_y + (rel_y * window_height)

    return int(abs_x), int(abs_y)


def click_at_coordinates(window_info: Optional[Dict[str, Any]], x: int, y: int, duration: float) -> bool:
    """Move mouse smoothly to coordinates and click using pyautogui."""
    try:
        import pyautogui

        pyautogui.FAILSAFE = True

        # Only focus window if window_info is provided
        if window_info:
            script = f"""
                tell application "System Events"
                    tell process "{window_info['kCGWindowOwnerName']}"
                        set frontmost to true
                    end tell
                end tell
            """
            subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
            time.sleep(0.2)  # Brief pause to let window focus take effect

        # Perform the click
        pyautogui.moveTo(x, y, duration=duration)
        pyautogui.click()

        return True

    except Exception as e:
        logger.error(f"Failed to click: {str(e)}")
        return False


def send_keys_to_window(window_info, keys: list[str]) -> bool:
    """Send keystrokes to specific window using AppleScript."""
    try:
        if not window_info:
            logger.error("No window info stored")
            return False

        # Simple keystroke command for each key
        key_commands = "\n".join([f'keystroke "{key}"' for key in keys])

        script = f"""
            tell application "System Events"
                tell process "{window_info['kCGWindowOwnerName']}"
                    set frontmost to true
                    {key_commands}
                end tell
            end tell
        """

        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
        logger.info(f"AppleScript result: {result.stdout}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to send keystrokes: {e.stderr if e.stderr else str(e)}")
        return False
