"""Keyboard interaction tool."""

import subprocess
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

import pyautogui
from langchain.pydantic_v1 import BaseModel
from langchain.pydantic_v1 import Field
from langchain.tools import BaseTool

from pixelpilot.logger import setup_logger

logger = setup_logger(__name__)


class KeyboardInput(BaseModel):
    """Schema for keyboard input."""

    text: Optional[str] = Field(None, description="Text to type")
    keys: Optional[List[str]] = Field(None, description="Special keys to send")
    modifiers: Optional[List[Literal["command", "option", "control", "shift"]]] = Field(
        None, description="Modifier keys to hold while pressing keys"
    )
    window_info: Dict[str, Any] = Field(description="Window information for targeting")
    secure: bool = Field(default=False, description="Whether this is sensitive input (like passwords)")


class KeyboardTool(BaseTool):
    """Tool for keyboard control."""

    name = "keyboard"
    description = "Type text or send keyboard commands"
    args_schema = KeyboardInput

    def __init__(self):
        super().__init__()
        self.last_window = None
        pyautogui.FAILSAFE = True

    def _focus_window(self, window_info: Dict[str, Any]) -> bool:
        """Focus the target window."""
        try:
            script = f"""
                tell application "System Events"
                    tell process "{window_info['kCGWindowOwnerName']}"
                        set frontmost to true
                    end tell
                end tell
            """
            subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
            time.sleep(0.2)  # Brief pause to let window focus take effect
            return True
        except Exception as e:
            logger.error(f"Failed to focus window: {e}")
            return False

    def _send_keys(
        self, window_info: Dict[str, Any], keys: List[str], modifiers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Send keyboard commands using AppleScript."""
        try:
            if not self._focus_window(window_info):
                return {"success": False, "error": "Failed to focus window"}

            # Handle modifiers
            modifier_map = {"command": "command", "option": "option", "control": "control", "shift": "shift"}

            if modifiers:
                # Convert to AppleScript modifier format
                mod_str = " down, ".join(modifier_map[m] for m in modifiers if m in modifier_map)
                if mod_str:
                    mod_str += " down"
                key_commands = [f"key code {self._key_to_code(k)} using {{{mod_str}}}" for k in keys]
            else:
                # Simple keystrokes without modifiers
                key_commands = [f'keystroke "{k}"' for k in keys]

            script = f"""
                tell application "System Events"
                    tell process "{window_info['kCGWindowOwnerName']}"
                        {chr(10).join(key_commands)}
                    end tell
                end tell
            """

            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
            logger.info("Keys sent successfully")
            return {"success": True, "output": result.stdout}

        except Exception as e:
            logger.error(f"Failed to send keys: {e}")
            return {"success": False, "error": str(e)}

    def _type_text(self, window_info: Dict[str, Any], text: str, secure: bool = False) -> Dict[str, Any]:
        """Type text using pyautogui."""
        try:
            if not self._focus_window(window_info):
                return {"success": False, "error": "Failed to focus window"}

            if secure:
                logger.info("Typing secure input...")
            else:
                logger.info(f"Typing text: {text}")

            pyautogui.write(text, interval=0.1)  # Add small delay between keystrokes
            return {"success": True, "length": len(text)}

        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            return {"success": False, "error": str(e)}

    def _key_to_code(self, key: str) -> int:
        """Convert key name to AppleScript key code."""
        key_codes = {
            "tab": 48,
            "return": 36,
            "enter": 36,
            "space": 49,
            "left": 123,
            "right": 124,
            "up": 126,
            "down": 125,
            "escape": 53,
            "delete": 51,
            ".": 47,
            ",": 43,
        }
        return key_codes.get(key.lower(), ord(key[0]))

    def _run(
        self,
        window_info: Dict[str, Any],
        text: Optional[str] = None,
        keys: Optional[List[str]] = None,
        modifiers: Optional[List[str]] = None,
        secure: bool = False,
    ) -> Dict[str, Any]:
        """Execute a keyboard action.

        Args:
            window_info: Window information for targeting
            text: Text to type
            keys: Special keys to send
            modifiers: Modifier keys to hold while pressing keys
            secure: Whether this is sensitive input

        Returns:
            Dict containing action results
        """
        # Validate window info
        if not window_info:
            return {"success": False, "error": "No window info provided"}

        # Store window info
        self.last_window = window_info

        # Handle text input
        if text is not None:
            return self._type_text(window_info, text, secure)

        # Handle key commands
        elif keys:
            return self._send_keys(window_info, keys, modifiers)

        else:
            return {"success": False, "error": "No text or keys provided"}
