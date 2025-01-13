from typing import List
from typing import Optional
from typing import TypedDict

from PIL import Image

from pixelpilot.models import Action


class SharedState(TypedDict):
    """Unified state for both terminal and visual operations."""

    # Task-specific fields
    task_description: str  # Current task being executed
    task_status: str  # Current status of the task
    action_history: List[Action]  # History of actions taken

    # Common fields
    messages: List[dict]  # Chat messages
    current_path: str  # "terminal" or "visual"
    context: dict  # Shared context between paths
    window_info: dict  # Window information

    # Terminal-specific fields
    command_history: List[str]
    current_directory: str
    last_output: str

    # Visual-specific fields
    screenshot: Optional[Image.Image]
    labeled_img: Optional[str]  # base64 encoded image
    label_coordinates: Optional[dict]
    parsed_content_list: Optional[list]


class PathManager:
    """Manages state and path switching."""

    def __init__(self):
        self.state = SharedState(
            # Task fields
            task_description="",
            task_status="pending",
            action_history=[],
            # Common fields
            messages=[],
            current_path="terminal",
            context={},
            window_info={},
            # Terminal fields
            command_history=[],
            current_directory="",
            last_output="",
            # Visual fields
            screenshot=None,
            labeled_img=None,
            label_coordinates={},
            parsed_content_list=[],
        )

    def should_use_terminal(self, task_description: str) -> bool:
        """Determine if a task should use the terminal path."""
        terminal_indicators = [
            "file",
            "directory",
            "command",
            "install",
            "run",
            "execute",
            "git",
            "process",
            "service",
            "disk",
            "create",
            "delete",
            "move",
            "copy",
        ]

        return any(indicator in task_description.lower() for indicator in terminal_indicators)

    def update_state(self, updates: dict) -> None:
        """Update state with new values."""
        for key, value in updates.items():
            if key in self.state:
                self.state[key] = value

    def switch_path(self, new_path: str) -> None:
        """Switch the current execution path."""
        self.state["current_path"] = new_path

    def get_state(self) -> SharedState:
        """Get the current state."""
        return self.state

    def add_action(self, action: Action) -> None:
        """Add an action to the history."""
        self.state["action_history"].append(action)
