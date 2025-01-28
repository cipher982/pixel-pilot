"""System Control interface for PixelPilot."""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from PIL import Image

from pixelpilot.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class OperationResult:
    """Result of any system operation."""

    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class SystemController(ABC):
    """Controls both GUI and terminal operations on a system."""

    @abstractmethod
    def setup(self) -> OperationResult:
        """Initialize the system controller."""
        pass

    # GUI Operations
    @abstractmethod
    def capture_screen(self) -> Tuple[Optional[Image.Image], OperationResult]:
        """Capture current screen state."""
        pass

    @abstractmethod
    def click(self, x: int, y: int) -> OperationResult:
        """Click at given coordinates."""
        pass

    @abstractmethod
    def type_text(self, text: str) -> OperationResult:
        """Type text at current cursor position."""
        pass

    # Terminal Operations
    @abstractmethod
    def run_command(self, command: str, **kwargs) -> OperationResult:
        """Run a terminal command.

        Args:
            command: The command to execute
            **kwargs: Additional arguments for command execution
                cwd: Working directory
                env: Environment variables
                timeout: Command timeout
        """
        pass

    @abstractmethod
    def get_current_directory(self) -> str:
        """Get current working directory."""
        pass

    @abstractmethod
    def set_current_directory(self, path: str) -> OperationResult:
        """Set current working directory."""
        pass

    # System State
    @abstractmethod
    def get_system_info(self) -> Dict[str, str]:
        """Get information about the system.

        Returns:
            Dict containing:
                os_type: Operating system type
                os_version: OS version
                python_version: Python version
                arch: System architecture
                shell: Current shell
        """
        pass

    # Lifecycle
    @abstractmethod
    def pause(self) -> OperationResult:
        """Pause/suspend the system."""
        pass

    @abstractmethod
    def resume(self) -> OperationResult:
        """Resume the system."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
