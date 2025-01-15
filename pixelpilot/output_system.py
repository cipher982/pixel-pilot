from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


class OutputType(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class TaskMetadata(BaseModel):
    """Metadata about task execution"""

    start_time: datetime
    end_time: Optional[datetime] = None
    total_steps: int = 0
    models_used: List[str] = []
    path_transitions: List[str] = []
    confidence: float = 0.0
    token_usage: Dict[str, int] = {}


class AITaskResult(BaseModel):
    """Task result with metadata"""

    type: OutputType
    message: str
    metadata: TaskMetadata


def create_task_result(state: Dict[str, Any], task_description: str) -> AITaskResult:
    """Create task result with metadata from state"""
    # Try to get message from different sources in order of preference
    message = state.get("summary")  # First try direct summary
    if not message:
        message = state.get("context", {}).get("summary")  # Then try context summary
    if not message:
        message = state.get("context", {}).get("last_action_result", {}).get("output", "No output available")

    # Extract metadata from state
    context = state.get("context", {})
    metadata = TaskMetadata(
        start_time=context.get("start_time", datetime.now()),
        end_time=datetime.now(),
        total_steps=len(state.get("action_history", [])),
        models_used=context.get("models_used", []),
        path_transitions=context.get("path_transitions", []),
        confidence=context.get("confidence", 0.0),
        token_usage=context.get("token_usage", {}),
    )

    return AITaskResult(type=OutputType.SUCCESS, message=message, metadata=metadata)


def _create_metadata_table(metadata: TaskMetadata) -> Table:
    """Create a rich table for metadata display"""
    table = Table(show_header=False, box=None)

    duration = (metadata.end_time - metadata.start_time).total_seconds() if metadata.end_time else 0
    table.add_row("Duration", f"{duration:.2f}s")
    table.add_row("Steps", str(metadata.total_steps))
    table.add_row("Models", ", ".join(metadata.models_used))
    table.add_row("Paths", " → ".join(metadata.path_transitions))
    table.add_row("Confidence", f"{metadata.confidence*100:.1f}%")

    if metadata.token_usage:
        token_str = ", ".join(f"{k}: {v}" for k, v in metadata.token_usage.items())
        table.add_row("Tokens", token_str)

    return table


def display_result(result: AITaskResult) -> None:
    """Display the task result with rich formatting"""
    if result.type == OutputType.SUCCESS:
        # Main message panel
        text = Text(result.message, style="bold")
        main_panel = Panel(text, title="Task Result", border_style="blue", padding=(1, 2))

        # Metadata panel
        metadata_table = _create_metadata_table(result.metadata)
        metadata_panel = Panel(metadata_table, title="Execution Details", border_style="cyan")

        # Display both panels
        console.print("\n", main_panel)
        console.print(metadata_panel)
    else:
        text = Text(f"Error: {result.message}", style="bold red")
        panel = Panel(text, title="Error", border_style="red", padding=(1, 2))
        console.print("\n", panel)
