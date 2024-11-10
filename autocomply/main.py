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
        self.last_process_time = 0
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

        if not self.no_audio:
            try:
                self.audio_capture.start_capture()
            except Exception as e:
                logger.error(f"Error starting audio capture: {e}")
                return False

        return True

    def should_process(self) -> bool:
        """Check if enough time has passed to process next state."""
        current_time = time.time()
        if current_time - self.last_process_time >= Config.SCREENSHOT_INTERVAL:
            self.last_process_time = current_time
            return True
        return False

    def run(self) -> None:
        """Main loop of the agent."""
        if not self.setup():
            return

        try:
            logger.info("Agent started successfully, beginning main loop...")
            while True:
                if self.should_process():
                    screenshot = self.window_capture.capture_window(self.chrome_window)
                    audio_text = self.audio_capture.get_text() if self.audio_capture else None

                    if screenshot:
                        events = self.action_system.run(screenshot, audio_text)
                        for event in events:
                            logger.info(f"Event: {event}")

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
def main(no_audio: bool):
    agent = ChromeAgent(no_audio=no_audio)
    agent.run()


if __name__ == "__main__":
    main()
