import time

from autocomply.action_system import ActionSystem
from autocomply.audio_capture import AudioCapture
from autocomply.config import Config
from autocomply.window_capture import WindowCapture


class ChromeAgent:
    def __init__(self):
        self.window_capture = WindowCapture()
        self.audio_capture = AudioCapture()
        self.action_system = ActionSystem()
        self.chrome_window = None
        self.last_audio_time = 0
        self.last_screenshot_time = 0
        self.last_action_time = 0

    def setup(self) -> bool:
        """Setup the agent and ensure all components are ready."""
        print("\nPlease select the Chrome window you want to control...")
        self.chrome_window = self.window_capture.select_window_interactive()

        if not self.chrome_window:
            print("Error: No window selected")
            return False

        if self.chrome_window.get("kCGWindowOwnerName") != "Google Chrome":
            print("Error: Selected window is not Chrome")
            return False

        try:
            self.audio_capture.start_capture()
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            return False

        return True

    def run(self) -> None:
        """Main loop of the agent."""
        if not self.setup():
            return

        try:
            while True:
                current_time = time.time()

                # Capture screenshot if interval elapsed
                screenshot = None
                if current_time - self.last_screenshot_time >= Config.SCREENSHOT_INTERVAL:
                    screenshot = self.window_capture.capture_window(self.chrome_window)
                    self.last_screenshot_time = current_time

                # Get transcription if interval elapsed
                transcription = None
                if current_time - self.last_audio_time >= Config.AUDIO_INTERVAL:
                    transcription = self.audio_capture.get_transcription()
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
            print("\nStopping agent...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.audio_capture.stop_capture()


if __name__ == "__main__":
    agent = ChromeAgent()
    agent.run()
