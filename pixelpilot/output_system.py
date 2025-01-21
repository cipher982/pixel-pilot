import json
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

from pixelpilot.logger import setup_logger

console = Console()


class OutputType(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class TaskStep(BaseModel):
    """Individual task step information"""

    command: str
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    duration: Optional[float] = None


class TaskMetadata(BaseModel):
    """Metadata about task execution"""

    start_time: datetime
    end_time: Optional[datetime] = None
    total_steps: int = 0
    models_used: List[str] = []
    path_transitions: List[str] = []
    confidence: float = 0.0
    token_usage: Dict[str, int] = {}

    @property
    def duration(self) -> float:
        """Calculate duration in seconds"""
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()


class AITaskResult(BaseModel):
    """Task result with metadata"""

    type: OutputType
    message: str
    metadata: TaskMetadata
    steps: List[TaskStep] = []
    task_description: str

    def to_json(self) -> str:
        """Convert to JSON format with all execution details"""
        return json.dumps(
            {
                "task_result": {
                    "task": self.task_description,
                    "success": self.type == OutputType.SUCCESS,
                    "summary": self.message,
                    "steps": [step.model_dump() for step in self.steps],
                    "metadata": {
                        "duration_seconds": self.metadata.duration,
                        "total_steps": self.metadata.total_steps,
                        "models": self.metadata.models_used,
                        "path_flow": self.metadata.path_transitions,
                        "confidence": self.metadata.confidence,
                        "token_usage": self.metadata.token_usage,
                    },
                }
            },
            indent=2,
        )


def create_task_result(state: Dict[str, Any], task_description: str) -> AITaskResult:
    """Create task result with metadata from state"""
    # Debug logging for state structure
    logger = setup_logger(__name__)
    logger.info("Full state structure:")
    logger.info(f"State keys: {state.keys()}")
    logger.info(f"Context keys: {state.get('context', {}).keys()}")
    logger.info(f"Command history: {state.get('command_history', [])}")
    for i, cmd in enumerate(state.get("command_history", [])):
        logger.info(f"Action result {i}: {state.get('context', {}).get(f'action_result_{i}')}")

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
        end_time=context.get("end_time", datetime.now()),
        total_steps=len(state.get("command_history", [])),
        models_used=context.get("models_used", []),
        path_transitions=context.get("path_transitions", []),
        confidence=context.get("confidence", 0.0),
        token_usage=context.get("token_usage", {}),
    )

    # Debug logging
    logger.info(f"Command history: {state.get('command_history', [])}")
    logger.info(f"Context metadata: {context}")

    # Extract steps from command history
    steps = []
    command_history = state.get("command_history", [])

    for i, cmd in enumerate(command_history):
        # Try to get result from indexed history first
        result = context.get(f"action_result_{i}")

        # For the last command, also check last_action_result as backup
        if not result and i == len(command_history) - 1:
            result = context.get("last_action_result")

        # If still no result, create empty result
        if not result:
            result = {"success": True, "output": "", "error": None}

        # Extract output from multiple possible locations
        output = (
            result.get("output")
            or result.get("stdout")
            or context.get(f"output_{i}")
            or context.get("last_output")
            or ""
        ).strip()

        # Extract error similarly
        error = result.get("error") or result.get("stderr") or context.get(f"error_{i}") or None

        logger.info(f"Step {i}: cmd={cmd}, result={result}, output={output}")
        steps.append(
            TaskStep(
                command=cmd,
                success=result.get("success", True),
                output=output,
                error=error,
                duration=result.get("duration"),
            )
        )

    logger.info(f"Final steps list: {steps}")
    return AITaskResult(
        type=OutputType.SUCCESS, message=message, metadata=metadata, steps=steps, task_description=task_description
    )


def _create_metadata_table(metadata: TaskMetadata) -> Table:
    """Create a rich table for metadata display"""
    table = Table(show_header=False, box=None)

    duration = (metadata.end_time - metadata.start_time).total_seconds() if metadata.end_time else 0
    table.add_row("Duration", f"{duration:.2f}s")
    table.add_row("Steps", str(metadata.total_steps))
    table.add_row("Models", ", ".join(metadata.models_used))
    table.add_row("Paths", " → ".join(metadata.path_transitions))
    table.add_row("Confidence", f"{metadata.confidence*100:.1f}%")

    # if metadata.token_usage:
    #     token_str = ", ".join(f"{k}: {v}" for k, v in metadata.token_usage.items())
    #     table.add_row("Tokens", token_str)

    return table


def display_result(result: AITaskResult, output_format: str = "pretty") -> None:
    """Display task result in specified format."""
    if output_format == "json":
        # Write to eval result file
        with open("eval_result.json", "w") as f:
            f.write(result.to_json())

    if output_format == "pretty":
        if result.type == OutputType.SUCCESS:
            # Steps panel
            steps_table = Table(show_header=False, box=None)
            for step in result.steps:
                status = "✓" if step.success else "✗"
                cmd_text = f"[bold]{step.command}[/bold]"
                if step.output:
                    cmd_text += f"\n  └─ {step.output}"
                if step.error:
                    cmd_text += f"\n  └─ [red]Error: {step.error}[/red]"
                if step.duration is not None:
                    cmd_text += f" [dim]({step.duration:.2f}s)[/dim]"
                steps_table.add_row(f"[{'green' if step.success else 'red'}]{status}[/]", cmd_text)
            steps_panel = Panel(steps_table, title="Steps", border_style="blue")
            console.print("\n", steps_panel)

            # Main message panel
            text = Text(result.message, style="bold")
            main_panel = Panel(text, title="Task Result", border_style="blue", padding=(1, 2))

            # Metadata panel
            metadata_table = _create_metadata_table(result.metadata)
            metadata_panel = Panel(metadata_table, title="Execution Details", border_style="cyan")

            # Display panels
            console.print(main_panel)
            console.print(metadata_panel)
        else:
            text = Text(f"Error: {result.message}", style="bold red")
            panel = Panel(text, title="Error", border_style="red", padding=(1, 2))
            console.print("\n", panel)
