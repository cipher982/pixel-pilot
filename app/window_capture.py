import pygetwindow as gw
import mss
import time
from typing import Optional, Dict
import numpy as np

class WindowCapture:
	def __init__(self):
		self.sct = mss.mss()
		
	def get_chrome_window(self) -> Optional[gw.Window]:
		"""Get the Chrome window if it exists."""
		windows = gw.getWindowsWithTitle('Google Chrome')
		return windows[0] if windows else None
	
	def capture_window(self, window: gw.Window) -> np.ndarray:
		"""Capture the specified window and return as numpy array."""
		monitor = {
			"top": window.top,
			"left": window.left,
			"width": window.width,
			"height": window.height
		}
		screenshot = self.sct.grab(monitor)
		return np.array(screenshot)
	
	def __del__(self):
		self.sct.close()

if __name__ == "__main__":
	# Simple test
	capture = WindowCapture()
	chrome = capture.get_chrome_window()
	if chrome:
		print(f"Found Chrome window: {chrome.title}")
		screenshot = capture.capture_window(chrome)
		print(f"Captured screenshot shape: {screenshot.shape}")