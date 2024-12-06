from typing import Optional

import click

from pixelpilot.action_system import ActionSystem
from pixelpilot.logger import setup_logger

logger = setup_logger(__name__)


@click.command()
@click.option("--enable-audio", is_flag=True, help="Enable audio capture")
@click.option("--debug", is_flag=True, help="Run in debug mode using test image")
@click.option("--task-profile", "-t", help="Path to task profile YAML file")
@click.option("--instructions", "-i", help="Override task instructions")
@click.option("--label-boxes", is_flag=True, help="Label boxes")
@click.option("--use-chrome", is_flag=True, help="Use Chrome window for capture")
@click.option("--use-firefox", is_flag=True, help="Use Firefox window for capture")
def main(
    enable_audio: bool = False,
    debug: bool = False,
    task_profile: Optional[str] = None,
    instructions: Optional[str] = None,
    label_boxes: bool = False,
    use_chrome: bool = False,
    use_firefox: bool = False,
):
    action_system = ActionSystem(
        task_profile=task_profile,
        instructions=instructions,
        label_boxes=label_boxes,
        llm_provider="openai",
        # llm_provider="local",
        # llm_config={"url": "http://jelly:8080"},
        no_audio=not enable_audio,
        debug=debug,
        use_chrome=use_chrome,
        use_firefox=use_firefox,
    )
    action_system.run()


if __name__ == "__main__":
    main()
