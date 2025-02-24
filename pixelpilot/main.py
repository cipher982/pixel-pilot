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
@click.option(
    "--output-format",
    "-o",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
    help="Output format (pretty for human readable, json for machine parsing)",
)
@click.option("--gui-mode", is_flag=True, help="Enable GUI automation mode")
@click.option("--window-info", help="JSON string with window information for GUI tests")
@click.option(
    "--mode",
    type=click.Choice(["native", "docker", "scrapybara"]),
    help="Controller mode for system operations",
)
def main(
    enable_audio: bool = False,
    debug: bool = False,
    task_profile: Optional[str] = None,
    instructions: Optional[str] = None,
    label_boxes: bool = False,
    llm_provider: str = "openai",
    output_format: str = "pretty",
    gui_mode: bool = False,
    window_info: Optional[str] = None,
    mode: Optional[str] = None,
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
        graph_system = DualPathGraph(
            window_info=window_info,
            start_terminal=is_terminal,
            llm_provider=llm_provider,
            controller_mode=mode,
        )
        try:
            # Add task instructions to state
            graph_system.path_manager.update_state({"task_description": task_instructions})
            result = graph_system.run(task_description=task_instructions)

            # Create and display task result
            state = {
                "command_history": graph_system.path_manager.state.get("command_history", []),
                "action_history": graph_system.path_manager.state.get("action_history", []),
                "context": result.get("context", {}),  # Use context from result
                "status": result.get("status"),
                "summary": result.get("summary"),
            }
            task_result = create_task_result(state=state, task_description=task_instructions)
            display_result(task_result, output_format=output_format)
        finally:
            # Ensure controller is cleaned up
            graph_system.cleanup()

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
