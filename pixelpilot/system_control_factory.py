"""Factory for creating system controllers."""

import os
import platform
import subprocess
from typing import Optional

from pixelpilot.logger import setup_logger
from pixelpilot.system_control import OperationResult
from pixelpilot.system_control import SystemController
from pixelpilot.system_control_scrapybara import ScrapybaraController

logger = setup_logger(__name__)


class LegacyControllerAdapter(SystemController):
    """Adapts legacy GUI controllers to the new SystemController interface."""

    def __init__(self, gui_controller):
        """Initialize with a legacy GUI controller."""
        self.gui = gui_controller

    def setup(self) -> OperationResult:
        """Initialize the system controller."""
        return OperationResult(success=True, message="Legacy controller initialized")

    def capture_screen(self):
        """Delegate to GUI controller."""
        return self.gui.capture_screen()

    def click(self, x: int, y: int):
        """Delegate to GUI controller."""
        return self.gui.click(x, y)

    def type_text(self, text: str):
        """Delegate to GUI controller."""
        return self.gui.type_text(text)

    def run_command(self, command: str, **kwargs) -> OperationResult:
        """Run command using subprocess."""
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

    def get_system_info(self) -> dict[str, str]:
        """Get system information."""
        return {
            "os_type": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "arch": platform.machine(),
            "shell": os.environ.get("SHELL", "unknown"),
        }

    def pause(self) -> OperationResult:
        """Pause/suspend the system."""
        return OperationResult(success=True, message="Pause not implemented for legacy controller")

    def resume(self) -> OperationResult:
        """Resume the system."""
        return OperationResult(success=True, message="Resume not implemented for legacy controller")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.gui.cleanup()


class SystemControllerFactory:
    """Factory for creating system controllers."""

    @staticmethod
    def create(mode: Optional[str] = None, api_key: Optional[str] = None) -> SystemController:
        """Create a system controller.

        Args:
            mode: The controller mode to use. If None, will be determined from environment.
                 Valid values: "scrapybara", "docker", "native"
            api_key: Optional API key for Scrapybara

        Returns:
            An initialized SystemController instance
        """
        # If no mode specified, check environment
        if not mode:
            mode = os.environ.get("PIXELPILOT_MODE", "native")

        # Create appropriate controller
        if mode == "scrapybara":
            return ScrapybaraController(api_key=api_key)
        elif mode == "docker":
            from pixelpilot.gui_control_docker import DockerGUIController

            return LegacyControllerAdapter(DockerGUIController())
        else:
            from pixelpilot.gui_control_native import NativeGUIController

            return LegacyControllerAdapter(NativeGUIController())

    @staticmethod
    def detect_mode() -> str:
        """Detect which mode should be used based on environment."""
        if os.environ.get("PIXELPILOT_MODE"):
            return os.environ["PIXELPILOT_MODE"]
        elif os.path.exists("/.dockerenv"):
            return "docker"
        else:
            return "native"
