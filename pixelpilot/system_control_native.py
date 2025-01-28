"""Native system controller implementation."""

import os
import platform
import subprocess
from typing import Dict
from typing import Optional
from typing import Tuple

import pyautogui
from PIL import Image

from pixelpilot.logger import setup_logger
from pixelpilot.system_control import OperationResult
from pixelpilot.system_control import SystemController

logger = setup_logger(__name__)


class NativeSystemController(SystemController):
    """Native system controller using PyAutoGUI for GUI and subprocess for terminal."""

    def __init__(self):
        """Initialize controller."""
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = 0.1  # Small delay between actions

    def setup(self) -> OperationResult:
        """Initialize the system controller."""
        try:
            # Test PyAutoGUI by getting screen size
            width, height = pyautogui.size()
            return OperationResult(
                success=True,
                message="Native controller initialized",
                details={"screen_width": width, "screen_height": height},
            )
        except Exception as e:
            logger.error(f"Failed to initialize native controller: {e}")
            return OperationResult(success=False, message=str(e))

    # GUI Operations
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
            return None, OperationResult(success=False, message=str(e))

    def click(self, x: int, y: int) -> OperationResult:
        """Click at given coordinates."""
        try:
            pyautogui.click(x=x, y=y)
            return OperationResult(success=True, message=f"Clicked at ({x}, {y})")
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return OperationResult(success=False, message=str(e))

    def type_text(self, text: str) -> OperationResult:
        """Type text at current cursor position."""
        try:
            pyautogui.typewrite(text)
            return OperationResult(success=True, message=f"Typed text: {text}")
        except Exception as e:
            logger.error(f"Type text failed: {e}")
            return OperationResult(success=False, message=str(e))

    # Terminal Operations
    def run_command(self, command: str, **kwargs) -> OperationResult:
        """Run a terminal command."""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, **kwargs)
            return OperationResult(
                success=result.returncode == 0,
                message=result.stdout if result.returncode == 0 else result.stderr,
                details={"returncode": result.returncode},
            )
        except Exception as e:
            return OperationResult(success=False, message=str(e))

    def get_current_directory(self) -> str:
        """Get current working directory."""
        return os.getcwd()

    def set_current_directory(self, path: str) -> OperationResult:
        """Set current working directory."""
        try:
            os.chdir(path)
            return OperationResult(success=True, message=f"Changed directory to {path}")
        except Exception as e:
            return OperationResult(success=False, message=str(e))

    # System State
    def get_system_info(self) -> Dict[str, str]:
        """Get information about the system."""
        return {
            "os_type": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "arch": platform.machine(),
            "shell": os.environ.get("SHELL", "unknown"),
        }

    # Lifecycle
    def pause(self) -> OperationResult:
        """Pause/suspend the system."""
        return OperationResult(success=True, message="Pause not implemented for native controller")

    def resume(self) -> OperationResult:
        """Resume the system."""
        return OperationResult(success=True, message="Resume not implemented for native controller")

    def cleanup(self) -> None:
        """Clean up resources."""
        # PyAutoGUI doesn't need cleanup
        pass

    # File Operations
    def file_exists(self, path: str) -> bool:
        """Check if a file exists in the controlled environment."""
        return os.path.isfile(path)

    def read_file(self, path: str) -> str:
        """Read contents of a file in the controlled environment."""
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return ""

    def get_file_size(self, path: str) -> int:
        """Get size of a file in bytes."""
        try:
            return os.path.getsize(path)
        except Exception as e:
            logger.error(f"Failed to get file size: {e}")
            return 0
