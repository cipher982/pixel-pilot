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

from autocomply.logger import setup_logger

logger = setup_logger(__name__)


MODEL_NAME = "gpt-4o-mini"


class State(TypedDict):
    """Define the state schema and reducers."""

    messages: Annotated[list, add_messages]
    screenshot: Optional[Image.Image]
    action: Optional[str]
    parameters: dict
    context: dict


class ActionSystem:
    def __init__(self, task_profile: Optional[str] = None, instructions: Optional[str] = None):
        self.llm = ChatOpenAI(model=MODEL_NAME)
        self.config = self._load_task_config(task_profile, instructions)

        # Initialize state
        self.current_state = {
            "messages": [],
            "context": {
                "last_action": None,
                "window_info": None,
            },
            "parameters": {},
        }

        self.graph = self._build_graph()
        logger.info("Action system initialized")

    def initialize(self) -> bool:
        """Initialize window capture and selection."""
        self.window_info = self.window_capture.select_window_interactive()
        if self.window_info:
            # Update window info in current state
            self.current_state["context"]["window_info"] = self.window_info
            return True
        return False

    def _load_task_config(self, profile_path: Optional[str], override: Optional[str]) -> dict:
        default_config = {
            "instructions": "Default instructions for general navigation",
            "actions": {
                "next_slide": {
                    "keys": ["tab"],
                    "description": "Move to next element",
                    "triggers": ["next element", "continue"],
                },
                "confirm": {"keys": ["enter"], "description": "Select element", "triggers": ["select", "confirm"]},
                "wait": {"keys": [], "description": "Wait for content", "triggers": ["loading", "playing"]},
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

    def set_window_info(self, window_info: dict) -> None:
        """Set window info in context."""
        self.current_state["context"]["window_info"] = window_info

    def should_decide(self, state: State) -> str:
        """Determine if we should move to decide step."""
        logger.info(f"should_decide: action={state.get('action')}")
        return "decide" if state.get("action") == "ANALYZED" else "analyze"

    def should_execute(self, state: State) -> str:
        """Determine if we should move to execute step."""
        logger.info(f"should_execute: action={state.get('action')}")

        if state.get("action") in ["WAIT", None]:
            return "end"  # Terminate the graph execution by routing to the 'end' node

        return "execute"

    def _encode_image(self, image: Image.Image) -> str:
        """Encode image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def analyze_inputs(self, state: State) -> dict:
        """Analyze screenshot and return updated state."""
        logger.info("Analyzing inputs...")

        messages = state["messages"]

        if state["screenshot"]:
            base64_image = self._encode_image(state["screenshot"])
            timestamp = time.strftime("%H:%M:%S")

            messages.append(
                HumanMessage(
                    content=[
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {
                            "type": "text",
                            "text": (
                                f"New screenshot taken at {timestamp}. "
                                "Focus on identifying any changes in: "
                                "1) Button states (enabled/disabled), "
                                "2) Video player status, "
                                "3) New content appearance. "
                                "What action should we take?"
                            ),
                        },
                    ]
                )
            )

            # Keep only last 2 screenshots to ensure we're comparing recent states
            if len(messages) > 3:
                messages = messages[-3:]

            return {"messages": messages, "action": "ANALYZED", "parameters": {}, "context": state["context"]}

        return {"messages": messages, "action": "WAIT", "parameters": {}, "context": state["context"]}

    def decide_action(self, state: State) -> dict:
        """Decide what action to take based on analysis."""
        # Only include trigger descriptions if triggers are defined
        actions_description = "\n".join(
            [f"- {name}: {details['description']}" for name, details in self.config["actions"].items()]
        )

        system_message = SystemMessage(
            content=dedent(f"""
                You are an automation assistant analyzing screenshots.
                Available actions:
                {actions_description}
                
                If content is loading or playing: Return WAIT
                Otherwise: Choose one of these actions: {', '.join(self.config['actions'].keys())}

                But, err on the side of actioning if you are unsure, cant hurt to try.
                
                Return a JSON object with:
                - "action": Either WAIT or one of the available actions
                - "description": Why you chose this action
            """).strip()
        )

        messages = [system_message] + state.get("messages", [])

        try:
            response = self.llm.invoke(messages, response_format={"type": "json_object"}).content
            response = json.loads(response)
            # logger.info(f"Decision: {response}")
            return {
                "messages": state["messages"],
                "action": response["action"],
                "parameters": response.get("parameters", {}),
                "context": state["context"],
            }
        except Exception as e:
            logger.error(f"Error in decide_action: {e}")
            return {"messages": state["messages"], "action": "WAIT"}

    def execute_action(self, state: State) -> dict:
        """Execute the decided action."""
        action = state["action"]
        context = state.get("context", {})

        logger.info(f"Executing action: {action}")

        if action in [None, "WAIT"]:
            logger.info("No action to execute.")
            return state  # Return the state as is

        try:
            action_config = self.config["actions"].get(action.lower())
            if not action_config:
                logger.error(f"Unknown action: {action}")
                return {"messages": [f"Unknown action {action}"], "action": "WAIT"}

            logger.info(f"Action config: {action_config}")  # Log the config

            if action_config["keys"]:
                window_info = context.get("window_info")
                logger.info(f"Window info for keystroke: {window_info}")  # Log window info

                if not window_info:
                    logger.error("No window info available")
                    return {"messages": ["No window info"], "action": "WAIT"}

                success = self._send_keys_to_window(action_config["keys"])
                if not success:
                    logger.error("Failed to send keystrokes")
                    return {"messages": ["Failed to send keystrokes"], "action": "WAIT"}

                logger.info(f"Keystrokes sent successfully: {'+'.join(action_config['keys'])}")
                time.sleep(1.5)
            else:  # WAIT action
                time.sleep(2)
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            return {"messages": [f"Failed to execute {action}"], "action": "WAIT"}

        context["last_action"] = action
        return {"messages": [f"Successfully executed {action}"], "action": None, "context": context}

    def _send_keys_to_window(self, keys: list[str]) -> bool:
        """Send keystrokes to specific window using window_id."""
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

            # Ensure there is at least one regular key to send
            if not regular_keys:
                logger.error("No regular keys to send")
                return False

            # For simplicity, send one regular key at a time
            # Modify as needed to handle multiple keys
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
            raise ValueError(f"Unknown key: {key}")  # Better error handling
        return code

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("analyze", self.analyze_inputs)
        workflow.add_node("decide", self.decide_action)
        workflow.add_node("execute", self.execute_action)
        # Add an 'end' node that does nothing
        workflow.add_node("end", lambda state: state)

        # Set entry point explicitly
        workflow.set_entry_point("analyze")

        # Linear flow with conditional termination
        workflow.add_edge("analyze", "decide")
        workflow.add_conditional_edges("decide", self.should_execute, {"execute": "execute", "end": "end"})

        return workflow.compile()

    def run(self, screenshot: Optional[Image.Image], audio_text: Optional[str]) -> list:
        """Run one iteration of the graph with current inputs."""
        # Update current state with new screenshot
        self.current_state = {
            "messages": self.current_state["messages"],
            "screenshot": screenshot,  # This gets lost in subsequent iterations
            "audio_text": audio_text,
            "action": None,
            "context": self.current_state["context"],
            "parameters": self.current_state["parameters"],
        }

        logger.info("Running action graph...")
        events = list(self.graph.stream(self.current_state))  # Use self.current_state instead
        logger.info(f"Graph generated {len(events)} events")
        return events
