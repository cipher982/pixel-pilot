import time

import click

from autocomply.action_system import ActionSystem
from autocomply.audio_capture import AudioCapture
from autocomply.config import Config
from autocomply.logger import setup_logger
from autocomply.window_capture import WindowCapture

logger = setup_logger(__name__)


class ChromeAgent:
    def __init__(self, no_audio: bool = False):
        self.window_capture = WindowCapture()
        self.audio_capture = None if no_audio else AudioCapture()
        self.action_system = ActionSystem()
        self.chrome_window = None
        self.last_audio_time = 0
        self.last_screenshot_time = 0
        self.last_action_time = 0
        self.no_audio = no_audio

    def setup(self) -> bool:
        """Setup the agent and ensure all components are ready."""
        logger.info("Please select the Chrome window you want to control...")
        self.chrome_window = self.window_capture.select_window_interactive()

        if not self.chrome_window:
            logger.error("No window selected")
            return False

        if self.chrome_window.get("kCGWindowOwnerName") != "Google Chrome":
            logger.error("Selected window is not Chrome")
            return False

        try:
            if not self.no_audio:
                self.audio_capture.start_capture()
        except Exception as e:
            logger.error(f"Error starting audio capture: {e}")
            return False

        return True

    def run(self) -> None:
        """Main loop of the agent."""
        if not self.setup():
            return

        try:
            logger.info("Agent started successfully, beginning main loop...")
            while True:
                current_time = time.time()

                # Capture screenshot if interval elapsed
                screenshot = None
                if current_time - self.last_screenshot_time >= Config.SCREENSHOT_INTERVAL:
                    logger.debug("Capturing screenshot...")
                    screenshot = self.window_capture.capture_window(self.chrome_window)
                    self.last_screenshot_time = current_time

                # Get transcription if interval elapsed
                transcription = None
                if current_time - self.last_audio_time >= Config.AUDIO_INTERVAL:
                    logger.debug("Getting audio transcription...")
                    transcription = self.audio_capture.get_transcription()
                    if transcription:
                        logger.info(f"Transcribed: {transcription}")
                    self.last_audio_time = current_time

                # Process action if we have new data and action interval elapsed
                if (screenshot or transcription) and current_time - self.last_action_time >= Config.ACTION_INTERVAL:
                    action = self.action_system.decide_action(transcription, screenshot)
                    if action:
                        action_type, coordinates = action
                        if action_type == "click":
                            self.action_system.click_at_position(coordinates)
                            self.last_action_time = current_time

                time.sleep(Config.MAIN_LOOP_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Stopping agent...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.audio_capture.stop_capture()


@click.command()
@click.option("--no-audio", is_flag=True, help="Disable audio capture")
def main(no_audio: bool):
    agent = ChromeAgent(no_audio=no_audio)
    agent.run()


if __name__ == "__main__":
    main()
