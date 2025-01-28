"""Scrapybara-based system controller implementation."""

import base64
import io
import platform
from typing import Dict
from typing import Optional
from typing import Tuple

from PIL import Image
from scrapybara import Scrapybara

from pixelpilot.logger import setup_logger
from pixelpilot.system_control import OperationResult
from pixelpilot.system_control import SystemController

logger = setup_logger(__name__)


class ScrapybaraController(SystemController):
    """Controls both GUI and terminal operations via Scrapybara VM."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize controller.

        Args:
            api_key: Optional Scrapybara API key. If not provided, will look for
                    SCRAPYBARA_API_KEY environment variable.
        """
        self.client = Scrapybara(api_key=api_key) if api_key else None
        self.instance = None
        self._current_directory = None  # Track CWD

    def setup(self) -> OperationResult:
        """Initialize the Scrapybara VM."""
        try:
            if not self.client:
                self.client = Scrapybara()
            self.instance = self.client.start_ubuntu()

            # Get initial working directory
            self._current_directory = self.instance.bash(command="pwd").strip()

            # Get system info
            system_info = {
                "os_type": "ubuntu",
                "os_version": self.instance.bash(command="lsb_release -rs").strip(),
                "python_version": self.instance.bash(command="python3 --version").strip(),
                "arch": self.instance.bash(command="uname -m").strip(),
                "shell": self.instance.bash(command="echo $SHELL").strip(),
            }

            return OperationResult(success=True, message="Scrapybara VM started successfully", details=system_info)
        except Exception as e:
            logger.error(f"Failed to start Scrapybara VM: {e}")
            return OperationResult(success=False, message=f"Failed to start VM: {str(e)}")

    # GUI Operations
    def capture_screen(self) -> Tuple[Optional[Image.Image], OperationResult]:
        """Capture current screen state."""
        if not self.instance:
            return None, OperationResult(success=False, message="VM not initialized")

        try:
            base64_img = self.instance.screenshot().base64_image
            # Convert base64 to PIL Image
            img = Image.open(io.BytesIO(base64.b64decode(base64_img)))
            return img, OperationResult(
                success=True, message="Screen captured successfully", details={"width": img.width, "height": img.height}
            )
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None, OperationResult(success=False, message=f"Screen capture failed: {str(e)}")

    def click(self, x: int, y: int) -> OperationResult:
        """Click at given coordinates."""
        if not self.instance:
            return OperationResult(success=False, message="VM not initialized")

        try:
            self.instance.computer(action="mouse_move", coordinate=[x, y])
            self.instance.computer(action="left_click")
            return OperationResult(success=True, message=f"Clicked at ({x}, {y})")
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return OperationResult(success=False, message=f"Click failed: {str(e)}")

    def type_text(self, text: str) -> OperationResult:
        """Type text at current cursor position."""
        if not self.instance:
            return OperationResult(success=False, message="VM not initialized")

        try:
            self.instance.computer(action="type", text=text)
            return OperationResult(success=True, message=f"Typed text: {text}")
        except Exception as e:
            logger.error(f"Type text failed: {e}")
            return OperationResult(success=False, message=f"Type text failed: {str(e)}")

    # Terminal Operations
    def run_command(self, command: str, **kwargs) -> OperationResult:
        """Run a terminal command."""
        if not self.instance:
            return OperationResult(success=False, message="VM not initialized")

        try:
            # Handle working directory if specified
            cwd = kwargs.get("cwd")
            if cwd:
                self.set_current_directory(cwd)

            # Run command
            output = self.instance.bash(command=command)
            return OperationResult(
                success=True, message="Command executed", details={"output": output, "cwd": self._current_directory}
            )
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return OperationResult(success=False, message=f"Command failed: {str(e)}")

    def get_current_directory(self) -> str:
        """Get current working directory."""
        if not self.instance:
            return ""

        if not self._current_directory:
            self._current_directory = self.instance.bash(command="pwd").strip()
        return self._current_directory

    def set_current_directory(self, path: str) -> OperationResult:
        """Set current working directory."""
        if not self.instance:
            return OperationResult(success=False, message="VM not initialized")

        try:
            # Try to cd and capture any errors
            result = self.instance.bash(command=f"cd {path} && pwd")
            self._current_directory = result.strip()
            return OperationResult(success=True, message=f"Changed directory to: {self._current_directory}")
        except Exception as e:
            return OperationResult(success=False, message=f"Failed to change directory: {str(e)}")

    def get_system_info(self) -> Dict[str, str]:
        """Get information about the system."""
        if not self.instance:
            return {
                "os_type": "ubuntu",
                "os_version": "22.04",  # Default from docs
                "python_version": platform.python_version(),
                "arch": platform.machine(),
                "shell": "/bin/bash",  # Default for Ubuntu
            }

        try:
            return {
                "os_type": "ubuntu",
                "os_version": self.instance.bash(command="lsb_release -rs").strip(),
                "python_version": self.instance.bash(command="python3 --version").strip(),
                "arch": self.instance.bash(command="uname -m").strip(),
                "shell": self.instance.bash(command="echo $SHELL").strip(),
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {
                "os_type": "ubuntu",
                "os_version": "22.04",
                "python_version": "unknown",
                "arch": "unknown",
                "shell": "unknown",
            }

    # Lifecycle
    def pause(self) -> OperationResult:
        """Pause/suspend the system."""
        if not self.instance:
            return OperationResult(success=False, message="VM not initialized")

        try:
            self.instance.pause()
            return OperationResult(success=True, message="VM paused successfully")
        except Exception as e:
            return OperationResult(success=False, message=f"Failed to pause VM: {str(e)}")

    def resume(self) -> OperationResult:
        """Resume the system."""
        if not self.instance:
            return OperationResult(success=False, message="VM not initialized")

        try:
            self.instance.resume()
            return OperationResult(success=True, message="VM resumed successfully")
        except Exception as e:
            return OperationResult(success=False, message=f"Failed to resume VM: {str(e)}")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.instance:
            try:
                self.instance.stop()
            except Exception as e:
                logger.error(f"Failed to stop VM: {e}")
        self.instance = None
        self._current_directory = None
