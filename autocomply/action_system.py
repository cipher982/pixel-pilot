import base64
import json
import subprocess
import time
from io import BytesIO
from textwrap import dedent
from typing import Annotated
from typing import Optional
from typing import TypedDict

import yaml
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from PIL import Image

from autocomply.audio_capture import AudioCapture
from autocomply.logger import setup_logger
from autocomply.window_capture import WindowCapture

logger = setup_logger(__name__)

MODEL_NAME = "gpt-4o-2024-08-06"


class State(TypedDict):
    """Define the state schema and reducers."""

    messages: Annotated[list, add_messages]
    screenshot: Optional[Image.Image]
    audio_text: Optional[str]
    action: Optional[str]
    parameters: dict
    context: dict


class ActionSystem:
    def __init__(
        self,
        task_profile: Optional[str] = None,
        instructions: Optional[str] = None,
        no_audio: bool = False,
        debug: bool = False,
    ):
        self.llm = ChatOpenAI(model=MODEL_NAME)
        self.config = self._load_task_config(task_profile, instructions)
        self.debug = debug

        # Initialize components
        self.window_capture = WindowCapture(debug=debug)
        self.audio_capture = None if (no_audio or debug) else AudioCapture()
        self.window_info = None  # Will be set during setup

        # Initialize state
        self.current_state = {
            "messages": [],
            "screenshot": None,
            "audio_text": None,
            "action": None,
            "parameters": {},
            "context": {
                "last_action": None,
                "window_info": None,
            },
        }

        # Initialize and compile the graph
        self.graph = self._build_graph()
        logger.info("Action system initialized")

    def setup(self) -> bool:
        """Setup the agent and ensure all components are ready."""
        if self.debug:
            return True

        logger.info("Please select the window you want to control...")
        self.window_info = self.window_capture.select_window_interactive()

        if not self.window_info:
            logger.error("No window selected")
            return False

        # Initialize action system with window info
        self.current_state["context"]["window_info"] = self.window_info
        logger.info(f"Window info stored: {self.window_info}")

        if self.audio_capture:
            try:
                self.audio_capture.start_capture()
            except Exception as e:
                logger.error(f"Error starting audio capture: {e}")
                return False

        return True

    def _load_task_config(self, profile_path: Optional[str], override: Optional[str]) -> dict:
        default_config = {
            "instructions": "Default instructions for general navigation",
            "actions": {
                "next_slide": {
                    "keys": ["tab"],
                    "description": "Move to next element",
                    "triggers": ["next element", "continue"],
                },
                "confirm": {
                    "keys": ["enter"],
                    "description": "Select element",
                    "triggers": ["select", "confirm"],
                },
                "wait": {
                    "keys": [],
                    "description": "Wait for content",
                    "triggers": ["loading", "playing"],
                },
            },
        }

        if override:
            return {"instructions": override, "actions": default_config["actions"]}

        if profile_path:
            with open(profile_path) as f:
                config = yaml.safe_load(f)
                # Ensure each action has triggers field
                for action in config["actions"].values():
                    action.setdefault("triggers", [])
                return config

        return default_config

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("capture_state", self.capture_state)
        workflow.add_node("analyze_inputs", self.analyze_inputs)
        workflow.add_node("decide_action", self.decide_action)
        workflow.add_node("execute_action", self.execute_action)
        workflow.add_node("end", lambda state: state)  # End node

        # Set entry point
        workflow.set_entry_point("capture_state")

        # Add edges
        workflow.add_edge("capture_state", "analyze_inputs")
        workflow.add_edge("analyze_inputs", "decide_action")

        # Conditional edge from decide_action
        workflow.add_conditional_edges(
            "decide_action", self.should_continue, {"execute_action": "execute_action", "end": "end"}
        )

        # Edge from execute_action back to capture_state
        workflow.add_edge("execute_action", "capture_state")

        return workflow.compile()

    def should_continue(self, state: State) -> str:
        """Determine whether to proceed to execute_action or end."""
        action = state.get("action")
        if action == "END":
            return "end"
        else:
            return "execute_action"

    def run(self) -> None:
        """Run the action system's graph."""
        if not self.setup():
            return

        logger.info("Starting action system graph...")
        try:
            self.graph.invoke(self.current_state)
        except KeyboardInterrupt:
            logger.info("Stopping action system...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.audio_capture:
            self.audio_capture.stop_capture()

    # Define graph nodes
    def capture_state(self, state: State) -> State:
        """Capture the current state (screenshot and audio)."""
        logger.info("Capturing state...")

        if self.debug:
            # Load test image
            screenshot = Image.open("test_image.png")
            audio_text = None
        else:
            screenshot = self.window_capture.capture_window(self.window_info)
            audio_text = self.audio_capture.get_transcription() if self.audio_capture else None

        state["screenshot"] = screenshot
        state["audio_text"] = audio_text
        return state

    def analyze_inputs(self, state: State) -> State:
        """Analyze screenshot and return updated state."""
        logger.info("Analyzing inputs...")

        messages = state.get("messages", [])

        if state["screenshot"]:
            base64_image = self._encode_image(state["screenshot"])
            timestamp = time.strftime("%H:%M:%S")

            messages.append(
                HumanMessage(
                    content=[
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {
                            "type": "text",
                            "text": dedent(f"""
                                New screenshot taken at {timestamp}.
                                Focus on identifying any changes in:
                                1) Button states (enabled/disabled),
                                2) Video player status,
                                3) New content appearance.
                                What action should we take?
                            """).strip(),
                        },
                    ]
                )
            )

            # Keep only last N messages to limit context size
            if len(messages) > 10:
                messages = messages[-10:]

            state["messages"] = messages
            state["action"] = "ANALYZED"
            return state

        else:
            logger.error("Failed to capture screenshot")
            state["action"] = "WAIT"
            return state

    def decide_action(self, state: State) -> State:
        """Decide what action to take based on analysis."""
        logger.info("Deciding action...")

        # Build the prompt
        actions_description = "\n".join(
            [f"- {name}: {details['description']}" for name, details in self.config["actions"].items()]
        )

        system_message = SystemMessage(
            content=dedent(f"""
                You are an automation assistant analyzing screenshots.
                Available actions:
                {actions_description}
                - END: Finish the task if it is complete.

                If the task is complete or no further actions are necessary, return 'END'.
                If content is loading or playing, return 'WAIT'.
                Otherwise, choose one of these actions: {', '.join(self.config['actions'].keys())}

                Return a JSON object with:
                - "action": One of the available actions including 'WAIT' or 'END'
                - "description": Why you chose this action
            """).strip()
        )

        messages = [system_message] + state.get("messages", [])

        try:
            response = self.llm.invoke(messages, response_format={"type": "json_object"}).content
            response = json.loads(response)
            logger.info(f"Decision: {response}")

            state["action"] = response["action"]
            state["parameters"] = response.get("parameters", {})
            return state

        except Exception as e:
            logger.error(f"Error in decide_action: {e}")
            state["action"] = "WAIT"
            return state

    def execute_action(self, state: State) -> State:
        """Execute the decided action."""
        action = state["action"]
        context = state.get("context", {})

        logger.info(f"Executing action: {action}")

        if action == "WAIT":
            logger.info("Action is WAIT, waiting for a while.")
            time.sleep(2)  # Adjust as needed
            # After waiting, reset action to None
            state["action"] = None
            return state

        if action == "END":
            logger.info("Action is END, task is complete.")
            # Optionally, perform any cleanup or final actions here.
            return state  # The graph will proceed to the 'end' node.

        if action is None:
            logger.info("No action to execute.")
            return state

        try:
            action_config = self.config["actions"].get(action.lower())
            if not action_config:
                logger.error(f"Unknown action: {action}")
                state["action"] = None
                return state

            if action_config["keys"]:
                window_info = context.get("window_info")
                if not window_info:
                    logger.error("No window info available")
                    state["action"] = None
                    return state

                success = self._send_keys_to_window(action_config["keys"])
                if not success:
                    logger.error("Failed to send keystrokes")
                    state["action"] = None
                    return state

                logger.info(f"Keystrokes sent successfully: {'+'.join(action_config['keys'])}")
                time.sleep(1.5)
            else:
                # Handle other actions that may not involve keys
                pass
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            state["action"] = None
            return state

        context["last_action"] = action
        state["context"] = context
        # After executing action, reset action to None
        state["action"] = None
        return state

    # Helper methods
    def _encode_image(self, image: Image.Image) -> str:
        """Encode image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _send_keys_to_window(self, keys: list[str]) -> bool:
        """Send keystrokes to specific window using AppleScript."""
        try:
            window_info = self.current_state["context"]["window_info"]
            if not window_info:
                logger.error("No window info stored")
                return False

            # Map our modifier names to AppleScript modifier syntax
            modifier_map = {
                "command": "command down",
                "option": "option down",
                "ctrl": "control down",
                "shift": "shift down",
            }

            modifiers = []
            regular_keys = []

            for key in keys:
                if key in modifier_map:
                    modifiers.append(modifier_map[key])
                else:
                    regular_keys.append(key)

            logger.info(f"Modifiers: {modifiers}")
            logger.info(f"Regular keys: {regular_keys}")

            # Build the using clause with modifiers
            if modifiers:
                using_clause = f" using {{{', '.join(modifiers)}}}"
            else:
                using_clause = ""

            if not regular_keys:
                logger.error("No regular keys to send")
                return False

            key_commands = "\n".join([f'keystroke "{key}"{using_clause}' for key in regular_keys])

            script = f"""
                tell application "System Events"
                    tell process "{window_info['kCGWindowOwnerName']}"
                        set frontmost to true
                        {key_commands}
                    end tell
                end tell
            """

            logger.info(f"Generated AppleScript:\n{script}")
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
            logger.info(f"AppleScript result: {result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to send keystrokes: {e.stderr if e.stderr else str(e)}")
            return False

    def _get_key_code(self, key: str) -> int:
        """Convert key name to AppleScript key code."""
        key_codes = {
            "command": 55,
            "option": 58,
            "ctrl": 59,
            ".": 47,
            "tab": 48,
            # Add more as needed
        }
        code = key_codes.get(key.lower())
        if code is None:
            raise ValueError(f"Unknown key: {key}")
        return code
