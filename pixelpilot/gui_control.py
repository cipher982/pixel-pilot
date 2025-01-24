"""GUI Control system for PixelPilot."""

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
    details: Dict[str, Any] = field(default_factory=dict)


class GUIController(ABC):
    """Simple GUI operations for testing."""

    @abstractmethod
    def capture_screen(self) -> Tuple[Optional[Image.Image], OperationResult]:
        """Capture current screen state.

        Returns:
            Tuple of (image, result). Image may be None if capture fails.
        """
        pass

    @abstractmethod
    def click(self, x: int, y: int) -> OperationResult:
        """Click at given coordinates."""
        pass

    @abstractmethod
    def type_text(self, text: str) -> OperationResult:
        """Type text at current cursor position."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
