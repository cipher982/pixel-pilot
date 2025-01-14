from enum import Enum
from typing import Any
from typing import Dict

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


class OutputType(str, Enum):
    SUCCESS = "success"  # Task completed successfully
    ERROR = "error"  # Task failed


class AITaskResult(BaseModel):
    """Simple task result with a clear message"""

    type: OutputType
    message: str


def create_task_result(state: Dict[str, Any], task_description: str) -> AITaskResult:
    """Create a simple task result focusing on the final answer"""
    # Get the AI-generated summary if available, otherwise use raw output
    message = state.get("context", {}).get("summary")
    if not message:
        message = state.get("context", {}).get("last_action_result", {}).get("output", "No output available")

    return AITaskResult(type=OutputType.SUCCESS, message=message)


def display_result(result: AITaskResult) -> None:
    """Display the task result with rich formatting"""
    if result.type == OutputType.SUCCESS:
        text = Text(result.message, style="bold")
        panel = Panel(text, title="Task Result", border_style="blue", padding=(1, 2))
        console.print("\n", panel)
    else:
        text = Text(f"Error: {result.message}", style="bold red")
        panel = Panel(text, title="Error", border_style="red", padding=(1, 2))
