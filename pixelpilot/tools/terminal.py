"""Terminal interaction tool."""

from typing import Optional

from pixelpilot.logger import setup_logger
from pixelpilot.models import Action
from pixelpilot.state_management import SharedState
from pixelpilot.system_control import SystemController
from pixelpilot.system_control_factory import SystemControllerFactory

logger = setup_logger(__name__)


class TerminalTool:
    """Tool for executing terminal commands."""

    def __init__(self, controller: Optional[SystemController] = None):
        """Initialize terminal tool with optional controller.

        Args:
            controller: SystemController to use for command execution.
                      If None, a native controller will be created.
        """
        self.controller = controller or SystemControllerFactory.create(mode="native")

    def execute_command(self, state: SharedState) -> SharedState:
        """Execute a terminal command."""
        action = Action(**state["context"]["next_action"])
        if action.type != "terminal":
            raise ValueError("Expected terminal action")

        command = action.command
        args = action.args or {}

        # Log the command being executed with any args
        if args:
            logger.info(f"Executing command: {command} (with args: {args})")
        else:
            logger.info(f"Executing command: {command}")

        try:
            from time import time

            start_time = time()

            # Use controller to run command
            result = self.controller.run_command(command, **args)
            duration = time() - start_time

            # Extract output and success from controller result
            success = result.success
            output = result.message if success else result.message
            error = None if success else result.message

            # Log command result
            if success:
                logger.info(f"Command succeeded with output: {output[:200]}{'...' if len(output) > 200 else ''}")
            else:
                logger.error(f"Command failed with error: {error}")

            # Track command history
            if "command_history" not in state:
                state["command_history"] = []
            current_cmd_index = len(state["command_history"])
            state["command_history"].append(command)

            # Add action to history
            if "action_history" not in state:
                state["action_history"] = []
            state["action_history"].append(action)

            # Store result both in last_action_result and in indexed history
            result_data = {
                "success": success,
                "output": output,
                "error": error,
                "duration": duration,
            }
            state["context"]["last_action_result"] = result_data
            state["context"][f"action_result_{current_cmd_index}"] = result_data
            state["last_output"] = output

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            error_data = {"success": False, "output": None, "error": str(e)}
            state["context"]["last_action_result"] = error_data
            if "command_history" in state:
                state["context"][f"action_result_{len(state['command_history'])-1}"] = error_data
            state["last_output"] = str(e)

        return state

    def analyze_output(self, state: SharedState) -> SharedState:
        """Pass through state for LLM to analyze."""
        return state
