import base64
import io
import json
import subprocess
import time
from io import BytesIO
from textwrap import dedent
from typing import Annotated
from typing import Optional
from typing import TypedDict

import yaml
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from PIL import Image

from autocomply.audio_capture import AudioCapture
from autocomply.logger import setup_logger
from autocomply.utils import check_ocr_box
from autocomply.utils import get_som_labeled_img
from autocomply.window_capture import WindowCapture

logger = setup_logger(__name__)

MODEL_NAME = "gpt-4o-2024-08-06"
MAX_MESSAGES = 5
USE_PARSER = False


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

        # Initialize placeholders for all models
        self._yolo_model = None
        self._florence_processor = None
        self._florence_model = None

        # # Initialize OmniParser models
        # try:
        #     logger.info("Loading OmniParser models...")
        #     self.yolo_model = YOLO("weights/icon_detect/best.pt")
        #     processor = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        #     model = AutoModelForCausalLM.from_pretrained("weights/icon_caption_florence", trust_remote_code=True)
        #     self.caption_model_processor = {"processor": processor, "model": model}
        #     logger.info("OmniParser models loaded successfully")
        # except Exception as e:
        #     logger.error(f"Failed to load OmniParser models: {e}")
        #     raise e

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

    @property
    def yolo_model(self):
        """Lazy load YOLO model only when needed"""
        if self._yolo_model is None:
            from ultralytics import YOLO

            logger.info("Loading YOLO model...")
            self._yolo_model = YOLO("weights/icon_detect/best.pt")
        return self._yolo_model

    @property
    def florence_models(self):
        """Lazy load Florence models only when needed"""
        if self._florence_processor is None or self._florence_model is None:
            from transformers import AutoModelForCausalLM

            logger.info("Loading Florence models...")

            self._florence_processor = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-base", trust_remote_code=True
            )
            self._florence_model = AutoModelForCausalLM.from_pretrained(
                "weights/icon_caption_florence", trust_remote_code=True
            )

        return {"processor": self._florence_processor, "model": self._florence_model}

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
        # workflow.add_node("analyze_inputs", self.analyze_inputs)
        workflow.add_node("decide_action", lambda state: self.decide_action(state, use_parser=USE_PARSER))
        workflow.add_node("execute_action", self.execute_action)
        workflow.add_node("end", lambda state: state)  # End node

        # Set entry point
        workflow.set_entry_point("capture_state")

        # Add edge
        workflow.add_edge("capture_state", "decide_action")

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
            logger.info("Stopping action system due to user interrupt...")
        except GraphRecursionError as e:
            logger.warning(f"Graph recursion limit reached: {e}")
            logger.info("Consider increasing recursion_limit if this is expected behavior")
        except Exception as e:
            logger.error(f"Unexpected error in action system: {e}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.audio_capture:
            self.audio_capture.stop_capture()

        # Force state to END to signal completion
        self.current_state["action"] = "END"
        self.current_state["messages"].append(SystemMessage(content="Session terminated by user."))

    def capture_state(self, state: State) -> State:
        """Capture the current state."""
        logger.info("Capturing state...")

        # Capture screenshot
        screenshot = self.window_capture.capture_window(self.window_info)
        state["screenshot"] = screenshot

        if USE_PARSER:
            logger.info("Processing with OmniParser...")
            img_buffer = BytesIO()
            screenshot.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            # Process screenshot with OmniParser
            box_threshold = 0.05  # Default value; can be configurable
            iou_threshold = 0.1  # Default value; can be configurable

            ocr_bbox_rslt, _ = check_ocr_box(
                img_buffer,
                display_img=False,
                output_bb_format="xyxy",
                goal_filtering=None,
                easyocr_args={"paragraph": False, "text_threshold": 0.9},
                use_paddleocr=False,
            )
            text, ocr_bbox = ocr_bbox_rslt

            dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                img_buffer,
                self.yolo_model,
                BOX_THRESHOLD=box_threshold,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config={
                    "text_scale": 0.8,
                    "text_thickness": 2,
                    "text_padding": 3,
                    "thickness": 3,
                },
                caption_model_processor=self.florence_models,
                ocr_text=text,
                iou_threshold=iou_threshold,
            )

            # Decode the labeled image
            labeled_image = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))
            state["labeled_image"] = labeled_image

            # Update state with parsed elements
            state["parsed_elements"] = {"parsed_content": parsed_content_list, "label_coordinates": label_coordinates}

            logger.info("State captured and processed with OmniParser")
        else:
            logger.info("State captured without OmniParser")

        return state

    def analyze_raw_image(self, state: State) -> State:
        """Analyze using raw screenshot with GPT-4V."""
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
                                What UI elements do you see and what action should we take?
                            """).strip(),
                        },
                    ]
                )
            )

            state["messages"] = messages[-MAX_MESSAGES:]
            return state
        return state

    def analyze_with_parser(self, state: State) -> State:
        """Analyze using OmniParser structured data."""
        messages = state.get("messages", [])
        parsed_elements = state.get("parsed_elements", {})

        if parsed_elements:
            messages.append(
                HumanMessage(
                    content=dedent(f"""
                        Detected UI elements:
                        {parsed_elements['parsed_content']}
                        
                        What action should we take based on these elements?
                    """).strip()
                )
            )

            state["messages"] = messages[-MAX_MESSAGES:]
            return state
        return state

    # def analyze_inputs(self, state: State) -> State:
    #     """Analyze screenshot and return updated state."""
    #     logger.info("Analyzing inputs...")

    #     messages = state.get("messages", [])

    #     if state["screenshot"]:
    #         base64_image = self._encode_image(state["screenshot"])
    #         timestamp = time.strftime("%H:%M:%S")

    #         messages.append(
    #             HumanMessage(
    #                 content=[
    #                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
    #                     {
    #                         "type": "text",
    #                         "text": dedent(f"""
    #                             New screenshot taken at {timestamp}.
    #                             Focus on identifying any changes in:
    #                             1) Button states (enabled/disabled),
    #                             2) Video player status,
    #                             3) New content appearance.
    #                             What action should we take?
    #                         """).strip(),
    #                     },
    #                 ]
    #             )
    #         )

    #         # Keep only last N messages to limit context size
    #         if len(messages) > MAX_MESSAGES:
    #             messages = messages[-MAX_MESSAGES:]

    #         state["messages"] = messages
    #         state["action"] = "ANALYZED"
    #         return state

    #     else:
    #         logger.error("Failed to capture screenshot")
    #         state["action"] = "WAIT"
    #         return state

    def decide_action(self, state: State, use_parser: bool = True) -> State:
        """Decide action based on specified analysis mode."""
        logger.info("Deciding action...")

        if use_parser:
            state = self.analyze_with_parser(state)
        else:
            state = self.analyze_raw_image(state)

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
                - "parameters": Required parameters for the chosen action
            """).strip()
        )

        messages = [system_message] + state.get("messages", [])

        try:
            response = self.llm.invoke(messages, response_format={"type": "json_object"}).content
            response = json.loads(response)
            logger.info(f"Decision: {response}")

            # Add AI's response to messages
            state["messages"].append(
                AIMessage(content=f"Decision: {response['description']} -> Action: {response['action']}")
            )

            state["action"] = response["action"]
            state["parameters"] = response.get("parameters", {})
            return state

        except Exception as e:
            logger.error(f"Error in decide_action: {e}")
            state["messages"].append(SystemMessage(content=f"Error occurred: {str(e)}. Defaulting to WAIT action."))
            state["action"] = "WAIT"
            return state

    def execute_action(self, state: State) -> State:
        """Execute the decided action."""
        action = state["action"]
        context = state.get("context", {})
        parameters = state.get("parameters", {})

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

                raw_keys = action_config["keys"]

                # Handle both dict and direct value parameters
                param_value = parameters.get("answer") if isinstance(parameters, dict) else parameters
                processed_keys = [str(param_value) if key == "${answer}" else key for key in raw_keys]

                logger.info(f"Sending keys: {processed_keys}")
                success = self._send_keys_to_window(processed_keys)
                if not success:
                    logger.error("Failed to send keystrokes")
                    state["action"] = None
                    return state

                logger.info(f"Keystrokes sent successfully: {'+'.join(processed_keys)}")
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

            # Simple keystroke command for each key
            key_commands = "\n".join([f'keystroke "{key}"' for key in keys])

            script = f"""
                tell application "System Events"
                    tell process "{window_info['kCGWindowOwnerName']}"
                        set frontmost to true
                        {key_commands}
                    end tell
                end tell
            """

            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
            logger.info(f"AppleScript result: {result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to send keystrokes: {e.stderr if e.stderr else str(e)}")
            return False

    def test_parser(self, image_path: str) -> None:
        """Simple test method to verify OmniParser integration."""
        try:
            # Use existing utils functions
            ocr_bbox_rslt, _ = check_ocr_box(
                image_path,
                display_img=False,
                output_bb_format="xyxy",
                goal_filtering=None,
                easyocr_args={"paragraph": False, "text_threshold": 0.9},
                use_paddleocr=False,
            )
            text, ocr_bbox = ocr_bbox_rslt

            # Get labeled image and parsed content
            labeled_img, coordinates, parsed_content = get_som_labeled_img(
                image_path,
                self.yolo_model,
                BOX_TRESHOLD=0.05,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                caption_model_processor=self.caption_model_processor,
                ocr_text=text,
                iou_threshold=0.1,
            )

            logger.info("Parser test successful!")
            logger.info(f"Parsed content: {parsed_content}")
            return True

        except Exception as e:
            logger.error(f"Parser test failed: {e}")
            return False
