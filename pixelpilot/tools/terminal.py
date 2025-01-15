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
        """Analyze command output and determine task completion."""
        result = state["context"].get("last_action_result", {})

        if not result.get("success", False):
            logger.error(f"Command failed: {result.get('error')}")
            return state

        # Check if task is complete based on output
        output = result.get("output", "")
        if self._is_task_complete(state["task_description"], output):
            logger.info("Task marked as completed in terminal tool")
            state["task_status"] = "completed"
            state["context"]["next_path"] = "end"

        return state

    def _is_task_complete(self, task: str, output: str) -> bool:
        """Check if the task is complete based on output."""
        task = task.lower()

        # Disk size task completion check
        if "disk" in task and "size" in task:
            has_disk_info = "filesystem" in output.lower() or "mounted" in output.lower()
            logger.info(f"Disk size task completion check: {has_disk_info}")
            return has_disk_info

        # Default to false for unknown tasks
        return False
