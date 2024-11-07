from window_capture import WindowCapture
from audio_capture import AudioCapture
from action_system import ActionSystem
import time
from typing import Optional
import sys

class ChromeAgent:
	def __init__(self):
		self.window_capture = WindowCapture()
		self.audio_capture = AudioCapture()
		self.action_system = ActionSystem()
		self.chrome_window = None
	
	def setup(self) -> bool:
		"""Setup the agent and ensure all components are ready."""
		self.chrome_window = self.window_capture.get_chrome_window()
		if not self.chrome_window:
			print("Error: Chrome window not found")
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
				transcription = self.audio_capture.get_transcription()
				
				# Decide and act
				action = self.action_system.decide_action(transcription, screenshot)
				if action:
					action_type, coordinates = action
					if action_type == "click":
						self.action_system.click_at_position(
							self.chrome_window, coordinates
						)
				
				# Small sleep to prevent excessive CPU usage
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