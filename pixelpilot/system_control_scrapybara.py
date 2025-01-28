"""Scrapybara-based system controller implementation."""

import base64
import io
import os
import platform
import time
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

from dotenv import load_dotenv
from PIL import Image
from scrapybara import Scrapybara

from pixelpilot.logger import setup_logger
from pixelpilot.system_control import OperationResult
from pixelpilot.system_control import SystemController

# Load environment variables
load_dotenv()

logger = setup_logger(__name__)


def _safe_strip(value: Union[str, Dict, None]) -> str:
    """Safely handle string operations on command output."""
    if isinstance(value, str):
        return value.strip()
    elif isinstance(value, dict):
        # Extract output from Scrapybara response format
        output = value.get("output", "")
        error = value.get("error", "")
        return (output + error).strip()
    return ""


class ScrapybaraController(SystemController):
    """Controls both GUI and terminal operations via Scrapybara VM."""

    def __init__(self):
        """Initialize controller."""
        api_key = os.getenv("SCRAPYBARA_API_KEY")
        if not api_key:
            raise ValueError("SCRAPYBARA_API_KEY environment variable not set")
        self.client = Scrapybara(api_key=api_key)
        self.instance = None
        self._current_directory = None  # Track CWD

    def setup(self) -> OperationResult:
        """Initialize the Scrapybara VM."""
        try:
            logger.info("Setting up Scrapybara controller...")
            if not self.client:
                logger.info("No client found, creating new one...")
                self.client = Scrapybara()

            logger.debug("Starting Ubuntu VM...")
            self.instance = self.client.start_ubuntu()
            logger.info("VM started successfully")
            time.sleep(2)  # Give VM time to fully initialize

            # Get initial working directory
            logger.info("Getting initial working directory...")
            self._current_directory = _safe_strip(self.instance.bash(command="pwd"))
            logger.info(f"Current directory: {self._current_directory}")
            time.sleep(0.5)  # Brief pause between commands

            # Get system info
            logger.debug("Getting system info...")
            system_info = {
                "os_type": "ubuntu",
                "os_version": _safe_strip(self.instance.bash(command="lsb_release -rs")),
                "python_version": _safe_strip(self.instance.bash(command="python3 --version")),
                "arch": _safe_strip(self.instance.bash(command="uname -m")),
                "shell": _safe_strip(self.instance.bash(command="echo $SHELL")),
            }
            logger.debug(f"System info: {system_info}")

            return OperationResult(success=True, message="Scrapybara VM started successfully", details=system_info)
        except Exception as e:
            logger.error(f"Failed to start Scrapybara VM: {e}")
            print(f"Error type: {type(e)}")
            print(f"Error attributes: {dir(e)}")
            self.cleanup()  # Ensure cleanup on failure
            return OperationResult(success=False, message=f"Failed to start VM: {str(e)}")

    def _ensure_vm_ready(self) -> bool:
        """Ensure VM is ready by checking a simple command."""
        if not self.instance:
            return False
        try:
            result = self.instance.bash(command="echo ready")
            return isinstance(result, dict) and result.get("output", "").strip() == "ready"
        except Exception:
            return False

    # GUI Operations
    def capture_screen(self) -> Tuple[Optional[Image.Image], OperationResult]:
        """Capture current screen state."""
        if not self.instance:
            return None, OperationResult(success=False, message="VM not initialized")

        try:
            screenshot = self.instance.screenshot()
            # Extract base64 image from response
            base64_img = screenshot.get("base64_image", "") if isinstance(screenshot, dict) else str(screenshot)

            if not base64_img:
                return None, OperationResult(success=False, message="Failed to get screenshot data")

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
        if not self.instance or not self._ensure_vm_ready():
            return OperationResult(success=False, message="VM not initialized or not responsive")

        try:
            # Handle working directory if specified
            cwd = kwargs.get("cwd")
            if cwd:
                self.set_current_directory(cwd)

            # Run command and extract output
            result = self.instance.bash(command=command)
            if isinstance(result, dict):
                output = result.get("output", "")
                error = result.get("error", "")
                success = not error
            else:
                output = str(result)
                error = ""
                success = True

            time.sleep(0.1)  # Brief pause between commands
            return OperationResult(
                success=success,
                message=output,
                details={"output": output, "error": error, "cwd": self._current_directory},
            )
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return OperationResult(success=False, message=f"Command failed: {str(e)}")

    def get_current_directory(self) -> str:
        """Get current working directory."""
        if not self.instance:
            return ""

        if not self._current_directory:
            self._current_directory = _safe_strip(self.instance.bash(command="pwd"))
        return self._current_directory

    def set_current_directory(self, path: str) -> OperationResult:
        """Set current working directory."""
        if not self.instance:
            return OperationResult(success=False, message="VM not initialized")

        try:
            # Try to cd and capture any errors
            result = self.instance.bash(command=f"cd {path} && pwd")
            self._current_directory = _safe_strip(result)
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
                "os_version": _safe_strip(self.instance.bash(command="lsb_release -rs")),
                "python_version": _safe_strip(self.instance.bash(command="python3 --version")),
                "arch": _safe_strip(self.instance.bash(command="uname -m")),
                "shell": _safe_strip(self.instance.bash(command="echo $SHELL")),
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
                logger.info("Stopping Scrapybara VM...")
                self.instance.stop()
                logger.info("VM stopped successfully")
                time.sleep(1)  # Give time for cleanup
            except Exception as e:
                logger.error(f"Failed to stop VM: {e}")
        self.instance = None
        self._current_directory = None

    # File Operations
    def file_exists(self, path: str) -> bool:
        """Check if a file exists in the controlled environment."""
        if not self.instance or not self._ensure_vm_ready():
            return False

        try:
            result = self.instance.bash(command=f"test -f {path} && echo 'true' || echo 'false'")
            return _safe_strip(result) == "true"
        except Exception as e:
            logger.error(f"Failed to check file existence: {e}")
            return False

    def read_file(self, path: str) -> str:
        """Read contents of a file in the controlled environment."""
        if not self.instance or not self._ensure_vm_ready():
            return ""

        try:
            result = self.instance.bash(command=f"cat {path}")
            return _safe_strip(result)
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return ""

    def get_file_size(self, path: str) -> int:
        """Get size of a file in bytes."""
        if not self.instance or not self._ensure_vm_ready():
            return 0

        try:
            result = self.instance.bash(command=f"stat -c %s {path}")
            size_str = _safe_strip(result)
            return int(size_str) if size_str.isdigit() else 0
        except Exception as e:
            logger.error(f"Failed to get file size: {e}")
            return 0
