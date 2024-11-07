import time

from action_system import ActionSystem
from audio_capture import AudioCapture
from window_capture import WindowCapture


class ChromeAgent:
    def __init__(self):
        self.window_capture = WindowCapture()
        self.audio_capture = AudioCapture()
        self.action_system = ActionSystem()
        self.chrome_window = None

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
                # Capture current state
                screenshot = self.window_capture.capture_window(self.chrome_window)
                if screenshot is None:
                    print("Failed to capture window")
                    continue

                transcription = self.audio_capture.get_transcription()

                # Decide and act
                action = self.action_system.decide_action(transcription, screenshot)
                if action:
                    action_type, coordinates = action
                    if action_type == "click":
                        self.action_system.click_at_position(coordinates)

                time.sleep(0.1)

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
