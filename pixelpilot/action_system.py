import base64
import subprocess
import time
from io import BytesIO
from textwrap import dedent
from typing import Annotated
from typing import List
from typing import Literal
from typing import Optional
from typing import TypedDict
from typing import Union

import numpy as np
import torch
import yaml
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from PIL import Image
from pydantic import BaseModel
from pydantic import Field

from pixelpilot.audio_capture import AudioCapture
from pixelpilot.logger import setup_logger
from pixelpilot.utils import log_runtime
from pixelpilot.window_capture import WindowCapture

logger = setup_logger(__name__)

MODEL_NAME = "gpt-4o-2024-08-06"
MAX_MESSAGES = 5
# USE_PARSER = False


class State(TypedDict):
    """Define the state schema and reducers."""

    messages: Annotated[list, add_messages]
    screenshot: Optional[Image.Image]
    audio_text: Optional[str]
    actions: list[dict]
    parameters: dict
    context: dict
    labeled_img: Optional[str]  # base64 encoded image
    label_coordinates: Optional[dict]
    parsed_content_list: Optional[list]


class ClickParameters(BaseModel):
    elementId: str


class ScrollParameters(BaseModel):
    """Optional parameters for scroll action"""

    amount: int = -400  # Negative values scroll down, positive up


class WaitParameters(BaseModel):
    duration: float = 2.0


class Action(BaseModel):
    action: Literal["click", "scroll", "wait", "end"]
    description: str
    parameters: Union[ClickParameters, WaitParameters, ScrollParameters, dict] = Field(default_factory=dict)


class ActionResponse(BaseModel):
    actions: List[Action]


class ActionSystem:
    def __init__(
        self,
        task_profile: Optional[str] = None,
        instructions: Optional[str] = None,
        no_audio: bool = False,
        debug: bool = False,
        use_parser: bool = False,
        enable_chains: bool = False,
        use_chrome: bool = False,
    ):
        # Set basic attributes first
        self.debug = debug
        self.use_parser = use_parser
        self.enable_chains = enable_chains
        self.use_chrome = use_chrome

        # Then load config and create models
        self.config = self._load_task_config(task_profile, instructions)
        self.llm = ChatOpenAI(model=MODEL_NAME).with_structured_output(ActionResponse)

        # Initialize placeholders for all models
        self._yolo_model = None
        self._florence_processor = None
        self._florence_model = None

        # Initialize device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Initialize components
        self.window_capture = WindowCapture(debug=debug)
        self.audio_capture = None if (no_audio or debug) else AudioCapture()
        self.window_info = None  # Will be set during setup

        # Initialize state
        self.current_state = {
            "messages": [],
            "screenshot": None,
            "audio_text": None,
            "actions": [],
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
            logger.info("YOLO model loaded successfully")
        return self._yolo_model

    @property
    def florence_models(self):
        """Lazy load Florence models only when needed"""
        if self._florence_processor is None or self._florence_model is None:
            from transformers import AutoModelForCausalLM
            from transformers import AutoProcessor

            logger.info("Loading Florence models...")

            self._florence_processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base", trust_remote_code=True
            )
            logger.info("Florence processor loaded successfully")
            self._florence_model = AutoModelForCausalLM.from_pretrained(
                "weights/icon_caption_florence", trust_remote_code=True
            ).to(self.device)
            logger.info("Florence caption model loaded successfully")
        return {"processor": self._florence_processor, "model": self._florence_model}

    def setup(self) -> bool:
        """Setup the agent and ensure all components are ready."""
        if self.debug:
            return True

        logger.info("Please select the window you want to control...")
        self.window_info = self.window_capture.select_window_interactive(use_chrome=self.use_chrome)

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
        """Load task configuration from YAML or use default/override."""
        default_config = {
            "instructions": "Default instructions for general navigation",
            "actions": {
                "click": "Click on a specific UI element (requires elementId)",
                "wait": "Wait for content to load (optional duration in seconds)",
                "end": "End the task when complete",
            },
        }

        if override:
            return {"instructions": override, "actions": default_config["actions"]}

        if profile_path:
            with open(profile_path) as f:
                return yaml.safe_load(f)

        return default_config

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("capture_state", self.capture_state)
        workflow.add_node("decide_action", lambda state: self.decide_action(state, use_parser=self.use_parser))
        workflow.add_node("execute_action", self.execute_action)

        # Set entry point
        workflow.set_entry_point("capture_state")

        # Add edges with END check
        workflow.add_edge("capture_state", "decide_action")
        workflow.add_conditional_edges(
            "decide_action",
            lambda state: "end"
            if any(a.get("action", "").lower() == "end" for a in state.get("actions", []))
            else "execute",
            {"end": END, "execute": "execute_action"},
        )
        workflow.add_edge("execute_action", "capture_state")

        return workflow.compile()

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
            logger.error(f"Unexpected error in action system: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.audio_capture:
            self.audio_capture.stop_capture()

        # Force state to END to signal completion
        self.current_state["action"] = "END"
        self.current_state["messages"].append(SystemMessage(content="Session terminated by user."))

    @log_runtime
    def capture_state(self, state: State) -> State:
        """Capture the current state."""
        logger.info("Capturing state...")

        # Capture screenshot
        screenshot = self.window_capture.capture_window(self.window_info)
        state["screenshot"] = screenshot

        if self.use_parser:
            from pixelpilot.utils import check_ocr_box
            from pixelpilot.utils import get_som_labeled_img

            logger.info("Processing with OmniParser...")

            # Convert PIL Image to numpy array once
            screenshot_np = np.array(screenshot)

            # Process with OCR
            logger.info("Checking OCR box...")
            ocr_bbox_rslt, _ = check_ocr_box(
                screenshot_np,  # Pass numpy array
                display_img=False,
                output_bb_format="xyxy",
                goal_filtering=None,
                easyocr_args={"paragraph": False, "text_threshold": 0.9},
                use_paddleocr=False,
            )
            text, ocr_bbox = ocr_bbox_rslt
            logger.info(f"OCR text: {text}")

            # Process with object detection
            box_threshold = 0.05
            iou_threshold = 0.1

            logger.info("Getting labeled image...")
            labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                screenshot_np,
                self.yolo_model,
                box_threshold=box_threshold,
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
            logger.info("Labeled image obtained successfully")
            state["labeled_img"] = labeled_img
            state["label_coordinates"] = label_coordinates
            state["parsed_content_list"] = parsed_content_list

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

    @log_runtime
    def analyze_with_parser(self, state: State) -> State:
        """Analyze using OmniParser structured data."""
        logger.info("Analyzing with parser...")
        messages = state.get("messages", [])
        parsed_content_list = state.get("parsed_content_list", [])
        labeled_img = state.get("labeled_img")

        if labeled_img and parsed_content_list:
            # Decode base64 string back to image, compress, then re-encode
            img_data = base64.b64decode(labeled_img)
            img = Image.open(BytesIO(img_data))

            # Get compressed base64 string
            compressed_base64 = self._encode_image(img)

            timestamp = time.strftime("%H:%M:%S")
            messages.append(
                HumanMessage(
                    content=[
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{compressed_base64}"}},
                        {
                            "type": "text",
                            "text": dedent(f"""
                                New screenshot taken at {timestamp}.
                                Detected UI elements:
                                {chr(10).join(parsed_content_list)}
                                
                                What action should we take based on these elements?
                            """).strip(),
                        },
                    ]
                )
            )
        else:
            raise ValueError("No labeled image or parsed content list found")

        logger.info("Analysis with parser complete")
        state["messages"] = messages[-MAX_MESSAGES:]
        return state

    @log_runtime
    def decide_action(self, state: State, use_parser: bool = True) -> State:
        """Decide action based on specified analysis mode."""
        logger.info("Deciding action...")

        if use_parser:
            state = self.analyze_with_parser(state)
        else:
            state = self.analyze_raw_image(state)

        # Build the prompt
        actions_description = "\n".join(
            [f"- {name}: {description}" for name, description in self.config["actions"].items()]
        )

        system_message = SystemMessage(
            content=dedent(f"""
                You are an automation assistant analyzing screenshots.
                
                Task Instructions:
                {self.config["instructions"]}
                
                Available actions:
                {actions_description}

                Return a JSON object with:
                - "actions": A list of actions, each with:
                - "action": One of the available actions
                - "description": Why you chose this action
                - "parameters": Required parameters for the chosen action
                    - For 'click': Must include 'elementId'
                    - For 'wait': Optional 'duration' in seconds (default: 2.0)
                    - For 'end': No parameters needed
            """).strip()
        )

        messages = [system_message] + state.get("messages", [])

        try:
            response = self.llm.invoke(messages)

            # Handle list of actions
            state["actions"] = [action.model_dump() for action in response.actions]
            logger.info(f"Decided actions: {state['actions']}")

            for action in response.actions:
                # Get parameters directly from the model_dump
                action_dict = action.model_dump()
                params_str = ", ".join(f"{k}={v}" for k, v in action_dict["parameters"].items())
                logger.info(f"Decided action: {action_dict['action']} ({params_str}) - {action_dict['description']}")

            actions_desc = "; ".join(f"{a.action}: {a.description}" for a in response.actions)
            state["messages"].append(AIMessage(content=f"Decisions: {actions_desc}"))

        except Exception as e:
            logger.error(f"Action decision failed: {str(e)}", exc_info=True)
            state["actions"] = [{"action": "end", "description": "Error encountered", "parameters": {}}]

        return state

    @log_runtime
    def execute_action(self, state: State) -> State:
        """Execute the decided action(s) with improved handling and feedback."""
        logger.info("Executing action(s)...")

        actions = state.get("actions", [])
        if not actions:
            logger.warning("No actions to execute")
            return state

        context = state.get("context", {})

        for action_obj in actions:
            # Access as dictionary since we stored it that way in decide_action
            action = action_obj["action"].lower()
            if action == "end":
                logger.info("END action received - terminating")
                state["actions"] = []
                state["messages"].append(SystemMessage(content="Task ended."))
                return state

            parameters = action_obj.get("parameters", {})
            logger.info(f"Executing action: {action} with parameters: {parameters}")

            # More detailed logging before execution
            if action == "click":
                element_id = parameters.get("elementId") or parameters.get("element_id")
                if element_id and element_id in state["label_coordinates"]:
                    parsed_content = state.get("parsed_content_list", [])
                    element_label = next(
                        (content for content in parsed_content if content.startswith(f"{element_id}:")),
                        f"Element {element_id}",
                    )
                    logger.info(f"Executing click on {element_label}")
                else:
                    logger.warning(f"Element ID {element_id} not found in coordinates")
            else:
                params_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
                logger.info(f"Executing {action} with {params_str}")

            try:
                match action:
                    case "click":
                        element_id = parameters.get("elementId") or parameters.get("element_id")
                        if not element_id or element_id not in state["label_coordinates"]:
                            raise ValueError(f"Invalid element_id: {element_id}")

                        rel_coords = state["label_coordinates"][element_id]
                        rel_x = rel_coords[0] + (rel_coords[2] / 2)
                        rel_y = rel_coords[1] + (rel_coords[3] / 2)
                        abs_x, abs_y = self._convert_relative_to_absolute(rel_x, rel_y)

                        time.sleep(0.5)
                        if not self._click_at_coordinates(abs_x, abs_y, duration=0.5):
                            raise RuntimeError("Click action failed")
                        time.sleep(0.5)

                    case "wait":
                        time.sleep(parameters.get("duration", 2.0))

                    case "scroll":
                        import pyautogui

                        # Focus window first
                        window_info = self.current_state["context"]["window_info"]
                        script = f"""
                            tell application "System Events"
                                tell process "{window_info['kCGWindowOwnerName']}"
                                    set frontmost to true
                                end tell
                            end tell
                        """
                        subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
                        time.sleep(0.2)

                        # Scroll down
                        amount = parameters.get("amount", -850)  # Use default if not specified
                        pyautogui.scroll(amount)  # Negative values scroll down
                        time.sleep(0.5)  # Wait for scroll to complete

                    case _:
                        raise ValueError(f"Unknown action: {action}")

                context["last_action"] = action
                logger.info(f"Action {action} completed successfully")

            except Exception as e:
                logger.error(f"Action execution failed: {str(e)}")
                context["last_error"] = str(e)
                break

        # Clean up state after all actions complete
        state["actions"] = []
        state["context"] = context

        logger.info("All actions completed successfully")
        return state

    # Helper methods
    def _convert_relative_to_absolute(self, rel_x: float, rel_y: float) -> tuple[int, int]:
        """Convert relative coordinates to absolute screen coordinates."""
        window_bounds = self.current_state["context"]["window_info"]["kCGWindowBounds"]
        window_x = window_bounds["X"]
        window_y = window_bounds["Y"]
        window_width = window_bounds["Width"]
        window_height = window_bounds["Height"]

        abs_x = window_x + (rel_x * window_width)
        abs_y = window_y + (rel_y * window_height)

        return int(abs_x), int(abs_y)

    def _click_at_coordinates(self, x: int, y: int, duration: float) -> bool:
        """Move mouse smoothly to coordinates and click using pyautogui."""
        try:
            import pyautogui

            pyautogui.FAILSAFE = True

            # First, ensure window is focused using AppleScript
            window_info = self.current_state["context"]["window_info"]
            script = f"""
                tell application "System Events"
                    tell process "{window_info['kCGWindowOwnerName']}"
                        set frontmost to true
                    end tell
                end tell
            """
            subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)

            # Brief pause to let window focus take effect
            time.sleep(0.2)

            # Now perform the click
            pyautogui.moveTo(x, y, duration=duration)
            pyautogui.click()

            logger.info(f"Clicked at ({x}, {y})")
            return True

        except Exception as e:
            logger.error(f"Failed to click: {str(e)}")
            return False

    def _encode_image(self, image: Image.Image) -> str:
        """Convert image to base64 string."""
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

    def test_parser(self, image_path: str) -> bool:
        """Simple test method to verify OmniParser integration."""
        from pixelpilot.utils import check_ocr_box
        from pixelpilot.utils import get_som_labeled_img

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
                box_threshold=0.05,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                caption_model_processor=self.florence_models["processor"],
                ocr_text=text,
                iou_threshold=0.1,
            )

            logger.info("Parser test successful!")
            logger.info(f"Parsed content: {parsed_content}")
            return True

        except Exception as e:
            logger.error(f"Parser test failed: {e}")
            return False
