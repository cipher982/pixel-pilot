"""Terminal interaction tool."""

from pixelpilot.logger import setup_logger
from pixelpilot.models import Action
from pixelpilot.state_management import SharedState

logger = setup_logger(__name__)


class TerminalTool:
    """Tool for executing terminal commands."""

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
            import subprocess
            from time import time

            start_time = time()
            result = subprocess.run(command, shell=True, capture_output=True, text=True, **args)
            duration = time() - start_time
            success = result.returncode == 0
            output = result.stdout if success else result.stderr

            # Log command result
            if success:
                logger.info(f"Command succeeded with output: {output[:200]}{'...' if len(output) > 200 else ''}")
            else:
                logger.error(f"Command failed with error: {result.stderr}")

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
                "error": result.stderr if not success else None,
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
