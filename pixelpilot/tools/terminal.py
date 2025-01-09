"""Terminal interaction tool."""

from typing import Any
from typing import Dict

from langchain.pydantic_v1 import BaseModel
from langchain.pydantic_v1 import Field
from langchain.tools import BaseTool


class TerminalInput(BaseModel):
    """Schema for terminal input."""

    command: str = Field(description="The command to execute")
    background: bool = Field(default=False, description="Run in background")


class TerminalTool(BaseTool):
    """Tool for executing terminal commands."""

    name = "terminal"
    description = "Execute terminal commands and get their output"
    args_schema = TerminalInput

    def _run(self, command: str, background: bool = False) -> Dict[str, Any]:
        """Execute a terminal command.

        Args:
            command: The command to execute
            background: Whether to run in background

        Returns:
            Dict containing execution results
        """
        # Implementation will be moved here later
        raise NotImplementedError("Terminal tool not implemented yet")
