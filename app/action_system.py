import pyautogui
from typing import Tuple, Optional
import pygetwindow as gw
import numpy as np

class ActionSystem:
	def __init__(self):
		# Ensure PyAutoGUI fails safe
		pyautogui.FAILSAFE = True
		
	def click_at_position(self, window: gw.Window, 
						 relative_pos: Tuple[int, int]) -> None:
		"""Click at a position relative to the window."""
		x = window.left + relative_pos[0]
		y = window.top + relative_pos[1]
		pyautogui.moveTo(x, y)
		pyautogui.click()
	
	def decide_action(self, transcribed_text: Optional[str], 
					 screenshot: Optional[np.ndarray]) -> Optional[Tuple[str, Tuple[int, int]]]:
		"""
		Simple decision system for MVP.
		Returns action type and coordinates if action needed.
		"""
		if not transcribed_text:
			return None
			
		# MVP: Just look for specific keywords
		transcribed_text = transcribed_text.lower()
		if "next" in transcribed_text or "continue" in transcribed_text:
			# For MVP, assume next button is always at this position
			return ("click", (100, 200))
			
		return None

if __name__ == "__main__":
	# Simple test
	action_sys = ActionSystem()
	test_window = gw.getWindowsWithTitle('Google Chrome')[0]
	
	# Test decision system
	action = action_sys.decide_action("Please click next", None)
	if action:
		print(f"Decided to perform action: {action}")
		if action[0] == "click":
			action_sys.click_at_position(test_window, action[1])