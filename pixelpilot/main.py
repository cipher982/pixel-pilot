from typing import Optional

import click

from pixelpilot.action_system import ActionSystem
from pixelpilot.logger import setup_logger

logger = setup_logger(__name__)


@click.command()
@click.option("--no-audio", is_flag=True, help="Disable audio capture")
@click.option("--debug", is_flag=True, help="Run in debug mode using test image")
@click.option("--task-profile", "-t", help="Path to task profile YAML file")
@click.option("--instructions", "-i", help="Override task instructions")
@click.option("--use-parser", is_flag=True, help="Use parser to analyze inputs")
@click.option("--enable-chains", is_flag=True, help="Enable multiple actions at once")
@click.option("--use-chrome", is_flag=True, help="Use Chrome window for capture")
def main(
    no_audio: bool = False,
    debug: bool = False,
    task_profile: Optional[str] = None,
    instructions: Optional[str] = None,
    use_parser: bool = False,
    enable_chains: bool = False,
    use_chrome: bool = False,
):
    action_system = ActionSystem(
        task_profile=task_profile,
        instructions=instructions,
        llm_provider="tgi",
        llm_config={"url": "http://jelly:8080"},
        no_audio=no_audio,
        debug=debug,
        use_parser=use_parser,
        enable_chains=enable_chains,
        use_chrome=use_chrome,
    )
    action_system.run()


if __name__ == "__main__":
    main()
