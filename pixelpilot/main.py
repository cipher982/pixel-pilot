from typing import Optional

import click
import yaml
from dotenv import load_dotenv

from pixelpilot.graph_system import DualPathGraph
from pixelpilot.logger import setup_logger
from pixelpilot.output_system import create_task_result
from pixelpilot.output_system import display_result
from pixelpilot.state_management import PathManager

logger = setup_logger(__name__)

load_dotenv()


@click.command()
@click.option("--enable-audio", is_flag=True, help="Enable audio capture")
@click.option("--debug", is_flag=True, help="Run in debug mode using test image")
@click.option("--task-profile", "-t", help="Path to task profile YAML file")
@click.option("--instructions", "-i", help="Override task instructions")
@click.option("--label-boxes", is_flag=True, help="Label boxes")
@click.option("--llm-provider", type=click.Choice(["local", "openai", "bedrock", "fireworks"]), default="openai")
def main(
    enable_audio: bool = False,
    debug: bool = False,
    task_profile: Optional[str] = None,
    instructions: Optional[str] = None,
    label_boxes: bool = False,
    llm_provider: str = "openai",
):
    """Main entry point for Pixel Pilot."""
    # Load task profile if provided
    task_instructions = instructions
    if task_profile:
        try:
            with open(task_profile, "r") as f:
                profile = yaml.safe_load(f)
                task_instructions = profile.get("instructions", instructions)
        except Exception as e:
            logger.error(f"Failed to load task profile: {e}")
            return

    if not task_instructions:
        logger.error("No task instructions provided")
        return

    # Initialize window capture
    window_info = None  # Always use full screen capture

    # Determine initial path and initialize system
    path_manager = PathManager()
    is_terminal = path_manager.should_use_terminal(task_instructions)
    logger.info(f"Starting in {'terminal' if is_terminal else 'visual'} mode")

    # Initialize and run the dual-path system
    try:
        graph_system = DualPathGraph(window_info=window_info, start_terminal=is_terminal)
        # Add task instructions to state
        graph_system.path_manager.update_state({"task_description": task_instructions})
        result = graph_system.run(task_description=task_instructions)

        # Create and display task result
        state = dict(graph_system.path_manager.state)
        state["status"] = result.get("status")  # Add result status to state
        task_result = create_task_result(state=state, task_description=task_instructions)
        display_result(task_result)

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
