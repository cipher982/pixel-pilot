"""Docker-based system controller implementation."""

import os
import platform
import re
import subprocess
from typing import Dict
from typing import Optional
from typing import Tuple

import Xlib.display
import Xlib.ext.xtest
import Xlib.X
from PIL import Image

from pixelpilot.logger import setup_logger
from pixelpilot.system_control import OperationResult
from pixelpilot.system_control import SystemController

logger = setup_logger(__name__)


def _normalize_display(display: str) -> str:
    """Convert XQuartz socket path to standard display format."""
    # Check if it's a Unix domain socket path (XQuartz style)
    if display.startswith("/"):
        # Extract display number from path
        match = re.search(r":(\d+)$", display)
        if match:
            return f":{match.group(1)}"
    return display


class DockerSystemController(SystemController):
    """Docker-based system controller using X11 for GUI and subprocess for terminal."""

    def __init__(self):
        """Initialize X11 connection."""
        display_name = _normalize_display(os.environ.get("DISPLAY", ":0"))
        self.display = Xlib.display.Display(display_name)
        self.screen = self.display.screen()
        self.root = self.screen.root

        # Test connection
        try:
            self.root.get_geometry()
            logger.info("Connected to X11 display")
        except Exception as e:
            logger.error(f"Failed to connect to X11: {e}")
            raise

    def setup(self) -> OperationResult:
        """Initialize the system controller."""
        try:
            # Get window geometry to test X11
            geom = self.root.get_geometry()
            return OperationResult(
                success=True,
                message="Docker controller initialized",
                details={"screen_width": geom.width, "screen_height": geom.height},
            )
        except Exception as e:
            logger.error(f"Failed to initialize Docker controller: {e}")
            return OperationResult(success=False, message=str(e))

    # GUI Operations
    def capture_screen(self) -> Tuple[Optional[Image.Image], OperationResult]:
        """Capture current screen state."""
        try:
            # Get window geometry
            geom = self.root.get_geometry()

            # Get raw image data
            raw = self.root.get_image(
                0,
                0,  # x, y
                geom.width,
                geom.height,
                Xlib.X.ZPixmap,
                0xFFFFFFFF,  # plane mask (all planes)
            )

            # Convert raw data to bytes if it's not already
            raw_data = raw.data
            if isinstance(raw_data, str):
                raw_data = raw_data.encode("latin-1")

            # Convert to PIL Image
            image = Image.frombytes(
                "RGB",
                (geom.width, geom.height),
                raw_data,
                "raw",
                "BGRX",  # X11 uses BGRX format
            )

            return image, OperationResult(
                success=True,
                message="Screen captured successfully",
                details={"width": geom.width, "height": geom.height},
            )
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None, OperationResult(success=False, message=str(e))

    def click(self, x: int, y: int) -> OperationResult:
        """Click at given coordinates."""
        try:
            # Move pointer
            self.display.screen().root.warp_pointer(x, y)
            self.display.sync()

            # Click
            Xlib.ext.xtest.fake_input(self.display, Xlib.X.ButtonPress, 1)
            self.display.sync()
            Xlib.ext.xtest.fake_input(self.display, Xlib.X.ButtonRelease, 1)
            self.display.sync()

            return OperationResult(success=True, message=f"Clicked at ({x}, {y})")
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return OperationResult(success=False, message=str(e))

    def type_text(self, text: str) -> OperationResult:
        """Type text at current cursor position."""
        try:
            # TODO: Implement with XTEST
            # This requires mapping characters to X11 keycodes
            return OperationResult(success=False, message="Type text not yet implemented")
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
        return OperationResult(success=True, message="Pause not implemented for Docker controller")

    def resume(self) -> OperationResult:
        """Resume the system."""
        return OperationResult(success=True, message="Resume not implemented for Docker controller")

    def cleanup(self) -> None:
        """Clean up X11 resources."""
        if hasattr(self, "display"):
            self.display.close()

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
