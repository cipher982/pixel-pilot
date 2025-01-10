"""Terminal interaction tool."""

import os
import shlex
import subprocess
from typing import Any
from typing import Dict

from langchain.pydantic_v1 import BaseModel
from langchain.pydantic_v1 import Field
from langchain.tools import BaseTool

from pixelpilot.logger import setup_logger

logger = setup_logger(__name__)


class TerminalInput(BaseModel):
    """Schema for terminal input."""

    command: str = Field(description="The command to execute")
    background: bool = Field(default=False, description="Run in background")


class TerminalTool(BaseTool):
    """Tool for executing terminal commands with persistent state."""

    name = "terminal"
    description = "Execute terminal commands and get their output"
    args_schema = TerminalInput

    def __init__(self):
        super().__init__()
        self.pwd = os.getcwd()
        self.env = os.environ.copy()
        self._shell_state = {"last_status": 0, "last_output": "", "history": []}

    def _run(self, command: str, background: bool = False) -> Dict[str, Any]:
        """Execute a terminal command maintaining shell state."""
        try:
            # Parse command properly using shlex
            args = shlex.split(command)

            # Handle cd command specially
            if args[0] == "cd":
                if len(args) > 1:
                    new_dir = os.path.expanduser(args[1])
                    new_dir = os.path.abspath(os.path.join(self.pwd, new_dir))
                    os.chdir(new_dir)
                    self.pwd = new_dir
                else:
                    # cd without args goes to home directory
                    self.pwd = os.path.expanduser("~")
                    os.chdir(self.pwd)

                logger.info(f"Changed directory to: {self.pwd}")
                return {"success": True, "output": "", "error": None, "pwd": self.pwd, "status": 0}

            # For all other commands
            logger.info(f"Executing command: {command}")
            result = subprocess.run(args, cwd=self.pwd, env=self.env, capture_output=True, text=True)

            # Update state
            self._shell_state["last_status"] = result.returncode
            self._shell_state["last_output"] = result.stdout
            self._shell_state["history"].append(command)

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "pwd": self.pwd,
                "status": result.returncode,
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            self._shell_state["last_status"] = e.returncode
            self._shell_state["last_output"] = e.stderr
            self._shell_state["history"].append(command)

            return {"success": False, "output": e.stderr, "error": str(e), "pwd": self.pwd, "status": e.returncode}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"success": False, "output": "", "error": str(e), "pwd": self.pwd, "status": 1}

    def get_state(self) -> Dict[str, Any]:
        """Get current shell state."""
        return {
            "pwd": self.pwd,
            "last_status": self._shell_state["last_status"],
            "last_output": self._shell_state["last_output"],
            "history": self._shell_state["history"],
        }
