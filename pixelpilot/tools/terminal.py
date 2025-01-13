"""Terminal interaction tool."""

import os
import shlex
import subprocess
import time
from typing import Any
from typing import Dict

from pixelpilot.logger import setup_logger
from pixelpilot.models import TerminalAction
from pixelpilot.state_management import SharedState

logger = setup_logger(__name__)


class TerminalTool:
    """Tool for executing terminal commands with persistent state."""

    def __init__(self):
        self._pwd = os.getcwd()
        self._env = os.environ.copy()

    def execute_command(self, state: SharedState) -> SharedState:
        """Execute the current command from state."""
        # Get the action from state
        action_data = state["context"].get("next_action")
        if not action_data:
            logger.error("No action found in state")
            return state

        action = TerminalAction(**action_data)
        result = self._run(action.command)

        # Update state with results
        state["current_directory"] = self._pwd
        state["last_output"] = result["output"]
        if result["error"]:
            state["last_output"] += f"\nError: {result['error']}"

        # Record action result
        state["context"]["last_action_result"] = {
            "success": result["success"],
            "command": action.command,
            "output": result["output"],
            "error": result["error"],
            "timestamp": time.time(),
        }

        state["command_history"].append(action.command)
        return state

    def _run(self, command: str) -> Dict[str, Any]:
        """Execute a terminal command maintaining shell state."""
        try:
            # Parse command properly using shlex
            args = shlex.split(command)
            if not args:
                return {"success": False, "output": "", "error": "Empty command", "pwd": self._pwd, "status": 1}

            # Handle cd command specially
            if args[0] == "cd":
                if len(args) > 1:
                    new_dir = os.path.expanduser(args[1])
                    new_dir = os.path.abspath(os.path.join(self._pwd, new_dir))
                    os.chdir(new_dir)
                    self._pwd = new_dir
                else:
                    # cd without args goes to home directory
                    self._pwd = os.path.expanduser("~")
                    os.chdir(self._pwd)

                logger.info(f"Changed directory to: {self._pwd}")
                return {"success": True, "output": "", "error": None, "pwd": self._pwd, "status": 0}

            # For all other commands
            logger.info(f"Executing command: {command}")
            result = subprocess.run(args, cwd=self._pwd, env=self._env, capture_output=True, text=True)

            return {
                "success": result.returncode == 0,
                "output": result.stdout.strip(),  # Strip whitespace
                "error": result.stderr.strip() if result.returncode != 0 else None,  # Strip whitespace
                "pwd": self._pwd,
                "status": result.returncode,
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            return {"success": False, "output": "", "error": str(e), "pwd": self._pwd, "status": e.returncode}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"success": False, "output": "", "error": str(e), "pwd": self._pwd, "status": 1}

    def analyze_output(self, state: SharedState) -> SharedState:
        """Analyze command output and determine next steps."""
        # Get the last action result
        last_result = state["context"].get("last_action_result", {})
        if not last_result:
            logger.warning("No action result to analyze")
            return state

        logger.info(f"Analyzing command result: success={last_result['success']}, output={last_result['output']}")

        # Update task status based on command success
        if last_result["success"]:
            state["task_status"] = "in_progress"

            # Check if the output satisfies the task
            is_complete = self._is_task_complete(state)
            logger.info(f"Task completion check: {is_complete}")

            if is_complete:
                state["task_status"] = "completed"
                state["context"]["task_completed"] = True
                logger.info("Task marked as completed in terminal tool")
        else:
            state["task_status"] = "failed"
            state["context"]["error"] = last_result["error"]
            logger.warning(f"Command failed with error: {last_result['error']}")

        logger.info(f"Final task status: {state['task_status']}")
        return state

    def _is_task_complete(self, state: SharedState) -> bool:
        """Check if the current task is complete based on output and task description."""
        task = state["task_description"].lower()
        output = state.get("last_output", "").lower()
        command_history = state.get("command_history", [])
        last_command = command_history[-1] if command_history else ""

        logger.debug(f"Checking task completion - Task: {task}")
        logger.debug(f"Last output: {output}")
        logger.debug(f"Last command: {last_command}")

        # Task-specific completion checks
        if "disk" in task and ("size" in task or "space" in task):
            has_size_info = any(term in output for term in ["gb", "mb", "tb", "bytes", "available", "capacity"])
            logger.info(f"Disk size task completion check: {has_size_info}")
            return has_size_info
        elif "create" in task and "file" in task:
            is_complete = "created" in output or (
                last_command.startswith("touch") and not state["context"].get("error")
            )
            logger.info(f"File creation task completion check: {is_complete}")
            return is_complete
        elif "install" in task:
            is_complete = "successfully installed" in output or "already installed" in output
            logger.info(f"Installation task completion check: {is_complete}")
            return is_complete

        # Default to not complete
        logger.info("No specific completion criteria matched, defaulting to not complete")
        return False
