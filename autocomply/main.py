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
    def __init__(self, no_audio: bool = False, debug: bool = False):
        self.debug = debug
        self.window_capture = WindowCapture(debug=debug)
        self.audio_capture = None if (no_audio or debug) else AudioCapture()
        self.action_system = ActionSystem()
        self.chrome_window = None
        self.last_capture_time = 0
        self.running = False

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

        if self.audio_capture:
            try:
                self.audio_capture.start_capture()
            except Exception as e:
                logger.error(f"Error starting audio capture: {e}")
                return False

        return True

    def capture_state(self) -> tuple[Optional[Image.Image], Optional[str]]:
        """Capture the current state (screenshot and audio)."""
        current_time = time.time()
        screenshot = None
        audio_text = None

        # Only capture new screenshot if enough time has passed
        if current_time - self.last_capture_time >= Config.SCREENSHOT_INTERVAL:
            screenshot = self.window_capture.capture_window(self.chrome_window)
            self.last_capture_time = current_time

        # Get latest audio text if available
        if self.audio_capture:
            audio_text = self.audio_capture.get_text()

        return screenshot, audio_text

    def process_events(self, events: list) -> None:
        """Process events returned from the action system."""
        if not events:  # Skip if no events
            return

        for event in events:
            if event.get("action") == "END":
                self.running = False
                logger.info("Received stop signal")
            logger.info(f"Event: {event}")

    def run(self) -> None:
        """Main loop of the agent."""
        if not self.setup():
            return

        try:
            logger.info("Agent started successfully, beginning main loop...")
            self.running = True

            # Initialize with first screenshot
            screenshot, _ = self.capture_state()
            if screenshot is None:
                logger.error("Failed to capture initial screenshot")
                return

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
def main(no_audio: bool, debug: bool):
    agent = ChromeAgent(no_audio=no_audio, debug=debug)
    agent.run()


if __name__ == "__main__":
    main()
