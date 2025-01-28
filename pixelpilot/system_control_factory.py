"""Factory for creating system controllers."""

import os
from typing import Optional

from pixelpilot.logger import setup_logger
from pixelpilot.system_control import SystemController
from pixelpilot.system_control_docker import DockerSystemController
from pixelpilot.system_control_native import NativeSystemController
from pixelpilot.system_control_scrapybara import ScrapybaraController

logger = setup_logger(__name__)


class SystemControllerFactory:
    """Factory for creating system controllers."""

    @staticmethod
    def create(mode: Optional[str] = None) -> SystemController:
        """Create a system controller."""
        # If no mode specified, check environment
        if not mode:
            mode = SystemControllerFactory.detect_mode()

        # Create appropriate controller
        if mode == "scrapybara":
            return ScrapybaraController()
        elif mode == "docker":
            return DockerSystemController()
        else:
            return NativeSystemController()

    @staticmethod
    def detect_mode() -> str:
        """Detect which mode should be used based on environment."""
        if os.environ.get("PIXELPILOT_MODE"):
            return os.environ["PIXELPILOT_MODE"]
        elif os.path.exists("/.dockerenv"):
            return "docker"
        else:
            return "native"
