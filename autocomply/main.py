import hashlib
import time
from typing import Optional

import click
from PIL import Image

from autocomply.action_system import ActionSystem
from autocomply.audio_capture import AudioCapture
from autocomply.config import Config
from autocomply.logger import setup_logger
from autocomply.window_capture import WindowCapture

logger = setup_logger(__name__)


class ChromeAgent:
    def __init__(
        self,
        no_audio: bool = False,
        debug: bool = False,
        task_profile: Optional[str] = None,
        instructions: Optional[str] = None,
    ):
        self.debug = debug
        self.window_capture = WindowCapture(debug=debug)
        self.audio_capture = None if (no_audio or debug) else AudioCapture()
        self.action_system = ActionSystem(task_profile=task_profile, instructions=instructions)
        self.chrome_window = None
        self.last_capture_time = -Config.SCREENSHOT_INTERVAL
        self.running = False
        self.last_screenshot_hash = None

    def setup(self) -> bool:
        """Setup the agent and ensure all components are ready."""
        if self.debug:
            return True

        logger.info("Please select the Chrome window you want to control...")
        self.chrome_window = self.window_capture.select_window_interactive()

        if not self.chrome_window:
            logger.error("No window selected")
            return False

        if self.chrome_window.get("kCGWindowOwnerName") != "Google Chrome":
            logger.error("Selected window is not Chrome")
            return False

        # Initialize action system with window info
        self.action_system.set_window_info(self.chrome_window)
        logger.info(f"Window info stored: {self.chrome_window}")

        if self.audio_capture:
            try:
                self.audio_capture.start_capture()
            except Exception as e:
                logger.error(f"Error starting audio capture: {e}")
                return False

        return True

    def capture_state(self) -> tuple[Optional[Image.Image], Optional[str]]:
        """Capture the current state (screenshot and audio)."""
        logger.info("Capturing state... (screenshot)")
        current_time = time.time()
        screenshot = None
        audio_text = None

        screenshot = self.window_capture.capture_window(self.chrome_window)
        self.last_capture_time = current_time

        if screenshot:
            # Track if screenshot has changed
            current_hash = hashlib.md5(screenshot.tobytes()).hexdigest()
            if current_hash != self.last_screenshot_hash:
                logger.info("New screenshot detected")
                self.last_screenshot_hash = current_hash
            else:
                logger.info("No new screenshot detected")
        else:
            logger.error("Failed to capture screenshot")

        if self.audio_capture:
            audio_text = self.audio_capture.get_transcription()

        logger.info("Returning screenshot")
        return screenshot, audio_text

    def process_events(self, events: list) -> None:
        """Process events returned from the action system."""
        if not events:  # Skip if no events
            return

        for event in events:
            if event.get("action") == "END":
                self.running = False
                logger.info("Received stop signal")
            # logger.info(f"Event: {event}")

    def run(self) -> None:
        """Main loop of the agent."""
        if not self.setup():
            return

        try:
            logger.info("Agent started successfully, beginning main loop...")
            self.running = True

            while self.running:
                # Capture current state
                screenshot, audio_text = self.capture_state()

                # Process new screenshot
                if screenshot is not None:
                    events = self.action_system.run(screenshot=screenshot, audio_text=audio_text)
                    self.process_events(events)
                else:
                    logger.error("Failed to capture screenshot")

                # Sleep to prevent excessive CPU usage
                time.sleep(Config.MAIN_LOOP_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Stopping agent...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.audio_capture:
            self.audio_capture.stop_capture()


@click.command()
@click.option("--no-audio", is_flag=True, help="Disable audio capture")
@click.option("--debug", is_flag=True, help="Run in debug mode using test image")
@click.option("--task-profile", "-t", help="Path to task profile YAML file")
@click.option("--instructions", "-i", help="Override task instructions")
def main(
    no_audio: bool = False,
    debug: bool = False,
    task_profile: Optional[str] = None,
    instructions: Optional[str] = None,
):
    agent = ChromeAgent(
        no_audio=no_audio,
        debug=debug,
        task_profile=task_profile,
        instructions=instructions,
    )
    agent.run()


if __name__ == "__main__":
    main()
