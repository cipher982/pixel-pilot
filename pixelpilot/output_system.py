from enum import Enum
from typing import Any
from typing import Dict

from pydantic import BaseModel
from rich.console import Console

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
    """Display the task result simply"""
    if result.type == OutputType.SUCCESS:
        console.print("\n" + result.message)
    else:
        console.print(f"\nError: {result.message}", style="red")
