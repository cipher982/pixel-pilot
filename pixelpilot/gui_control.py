"""GUI Control system for PixelPilot."""

import os
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
    """Result of a GUI operation."""

    success: bool
    message: str
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


class GUIController(ABC):
    """Abstract base class for GUI operations."""

    @abstractmethod
    def initialize(self) -> OperationResult:
        """Initialize the controller and verify it's working."""
        pass

    @abstractmethod
    def capture_screen(self) -> Tuple[Optional[Image.Image], OperationResult]:
        """Capture current screen state.

        Returns:
            Tuple of (image, result). Image may be None if capture fails.
        """
        pass

    @abstractmethod
    def click(self, x: int, y: int) -> OperationResult:
        """Perform mouse click at specified coordinates."""
        pass

    @abstractmethod
    def scroll(self, amount: int) -> OperationResult:
        """Scroll by given amount (positive = up, negative = down)."""
        pass

    @abstractmethod
    def get_screen_size(self) -> Tuple[int, int]:
        """Get current screen dimensions (width, height)."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources. Called on shutdown."""
        pass


class GUIControllerFactory:
    """Factory for creating appropriate GUIController instances."""

    @staticmethod
    def create(mode: str = "auto") -> GUIController:
        """Create a GUIController instance.

        Args:
            mode: One of "auto", "eval", or "host"

        Returns:
            Appropriate GUIController instance

        Raises:
            ValueError: If mode is invalid or required dependencies missing
        """
        if mode not in ["auto", "eval", "host"]:
            raise ValueError(f"Invalid mode: {mode}")

        # Auto-detect mode if not specified
        if mode == "auto":
            mode = "eval" if GUIControllerFactory._is_eval_env() else "host"

        if mode == "eval":
            # Lazy import to avoid loading X11 deps unnecessarily
            from pixelpilot.gui_control_eval import EvalGUIController

            return EvalGUIController()
        else:
            # TODO: Implement host-specific controllers
            raise NotImplementedError("Host mode not yet implemented")

    @staticmethod
    def _is_eval_env() -> bool:
        """Detect if we're running in eval environment."""
        # Check for our custom env var first
        if os.environ.get("PIXELPILOT_EVAL") == "1":
            return True

        # Fallback checks for Docker/X11
        in_docker = os.path.exists("/.dockerenv")
        has_x11 = bool(os.environ.get("DISPLAY"))

        return in_docker and has_x11
