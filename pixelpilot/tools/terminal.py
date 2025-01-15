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

        try:
            import subprocess

            result = subprocess.run(command, shell=True, capture_output=True, text=True, **args)
            success = result.returncode == 0
            output = result.stdout if success else result.stderr

            # Track command history
            if "command_history" not in state:
                state["command_history"] = []
            state["command_history"].append(command)

            state["context"]["last_action_result"] = {
                "success": success,
                "output": output,
                "error": result.stderr if not success else None,
            }
            state["last_output"] = output

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            state["context"]["last_action_result"] = {"success": False, "output": None, "error": str(e)}
            state["last_output"] = str(e)

        return state

    def analyze_output(self, state: SharedState) -> SharedState:
        """Pass through state for LLM to analyze."""
        return state
