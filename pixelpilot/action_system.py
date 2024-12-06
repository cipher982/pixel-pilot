import base64
import json
import subprocess
import time
from io import BytesIO
from textwrap import dedent
from typing import Annotated
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import TypedDict
from typing import Union

import numpy as np
import torch
import yaml
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from PIL import Image
from pydantic import BaseModel
from pydantic import Field

from pixelpilot.audio_capture import AudioCapture
from pixelpilot.logger import setup_logger
from pixelpilot.tgi_wrapper import LocalTGIChatModel
from pixelpilot.utils import get_som_labeled_img
from pixelpilot.utils import log_runtime
from pixelpilot.window_capture import WindowCapture

logger = setup_logger(__name__)

# Semantic models
CAPTION_MODEL = "florence"
# CAPTION_MODEL = "blip"
# ENABLE_FLORENCE_CAPABILITY = False
# DECISION_MODEL = "gpt-4o-2024-08-06"
MAX_MESSAGES = 10


class ClickParameters(BaseModel):
    elementId: str


class ScrollParameters(BaseModel):
    """Optional parameters for scroll action"""

    amount: int = -3400  # Negative values scroll down, positive up


class WaitParameters(BaseModel):
    """Optional parameters for wait action"""

    duration: float = 2.0


class BackParameters(BaseModel):
    """Optional parameters for back action"""

    pass


class Action(BaseModel):
    """Action schema"""

    description: str
    action: Literal["click", "scroll", "wait", "end", "back"]
    parameters: Union[ClickParameters, WaitParameters, ScrollParameters, BackParameters, dict] = Field(
        default_factory=dict
    )


class ActionResponse(BaseModel):
    actions: List[Action]


class State(TypedDict):
    """Define the state schema and reducers."""

    messages: Annotated[list, add_messages]
    screenshot: Optional[Image.Image]
    audio_text: Optional[str]
    actions: List[Action]
    parameters: dict
    context: dict
    labeled_img: Optional[str]  # base64 encoded image
    label_coordinates: Optional[dict]
    parsed_content_list: Optional[list]


class ActionSystem:
    """Action system for automating UI interactions."""

    def __init__(
        self,
        task_profile: Optional[str] = None,
        instructions: Optional[str] = None,
        llm_provider: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
        no_audio: bool = False,
        debug: bool = False,
        label_boxes: bool = False,
        use_chrome: bool = False,
        use_firefox: bool = False,
    ):
        """Initialize the action system."""
        self.debug = debug
        self.label_boxes = label_boxes
        self.use_chrome = use_chrome
        self.use_firefox = use_firefox

        # Then load config and create models
        self.config = self._load_task_config(task_profile, instructions)

        # Initialize LLM based on provider
        self.llm_provider = llm_provider
        self.llm_config = llm_config
        if llm_provider == "local":
            assert self.llm_config
            self.llm = LocalTGIChatModel(base_url=self.llm_config["url"], timeout=self.llm_config.get("timeout", 60.0))
        elif llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(model="gpt-4o").with_structured_output(ActionResponse)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        # Initialize placeholders for all models
        self._yolo_model = None
        self._florence_processor = None if self.label_boxes else False
        self._florence_model = None if self.label_boxes else False
        self._blip_processor = None
        self._blip_model = None

        # Initialize device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # self.device = "cpu"
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
            "label_coordinates": {},  # Initialize empty label coordinates
            "labeled_img": None,
            "parsed_content_list": [],
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
    def caption_model(self):
        """Get the currently selected caption model."""
        if CAPTION_MODEL == "blip":
            raise NotImplementedError("BLIP captioning is not yet implemented")
        return self.florence_models

    @property
    def florence_models(self):
        """Lazy load Florence models only when needed"""
        if not self.label_boxes:
            return None

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
        self.window_info = self.window_capture.select_window_interactive(
            use_chrome=self.use_chrome, use_firefox=self.use_firefox
        )

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
        workflow.add_node("decide_action", lambda state: self.decide_action(state, label_boxes=self.label_boxes))  # type: ignore
        workflow.add_node("execute_action", self.execute_action)

        # Set entry point
        workflow.set_entry_point("capture_state")

        # Add edges with END check
        workflow.add_edge("capture_state", "decide_action")
        workflow.add_conditional_edges(
            "decide_action",
            lambda state: "end" if any(a.action == "end" for a in state["actions"]) else "execute",
            {"end": END, "execute": "execute_action"},
        )
        workflow.add_edge("execute_action", "capture_state")

        return workflow.compile()  # type: ignore

    def run(self) -> None:
        """Run the action system's graph."""
        if not self.setup():
            return

        logger.info("Starting action system graph...")
        try:
            self.graph.invoke(self.current_state, {"recursion_limit": 100})  # type: ignore
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
        screenshot = self.window_capture.capture_window(self.window_info)  # type: ignore
        state["screenshot"] = screenshot

        if screenshot is None:
            return state

        # Convert PIL Image to numpy array once
        screenshot_np = np.array(screenshot)

        # Process with OCR if parser is enabled
        ocr_text = []
        ocr_bbox = None
        if self.label_boxes:
            from pixelpilot.utils import check_ocr_box

            logger.info("Processing with OCR...")
            ocr_bbox_rslt, _ = check_ocr_box(
                screenshot_np,
                display_img=False,
                output_bb_format="xyxy",
                goal_filtering=None,
                easyocr_args={"paragraph": False, "text_threshold": 0.9},
                use_paddleocr=False,
            )
            ocr_text, ocr_bbox = ocr_bbox_rslt
            logger.info(f"OCR text: {ocr_text}")

        # Run object detection with YOLO and optionally Florence
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
            caption_model_processor=self.caption_model if self.label_boxes else None,
            ocr_text=ocr_text,
            iou_threshold=iou_threshold,
        )
        logger.info("Labeled image obtained successfully")

        state["labeled_img"] = labeled_img
        state["label_coordinates"] = label_coordinates
        state["parsed_content_list"] = parsed_content_list if self.label_boxes else []

        return state

    def analyze_raw_image(self, state: State) -> State:
        """Analyze using raw screenshot with GPT-4V."""
        messages = state.get("messages", [])
        labeled_img = state.get("labeled_img")
        label_coordinates = state.get("label_coordinates", {})

        if labeled_img:
            # Save image to temp file
            img_data = base64.b64decode(labeled_img)
            img = Image.open(BytesIO(img_data))
            compressed_base64 = self._encode_image(img)

            timestamp = time.strftime("%H:%M:%S")

            # Create a list of available box IDs
            assert label_coordinates, "label_coordinates should not be empty"
            available_boxes = sorted(label_coordinates.keys())
            box_info = "\n".join([f"Box {box_id}" for box_id in available_boxes])

            messages.append(
                HumanMessage(
                    content=[
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{compressed_base64}"}},
                        {
                            "type": "text",
                            "text": dedent(f"""
                                New screenshot taken at {timestamp}.
                                Available UI elements:
                                {box_info}
                                
                                What action should we take based on these elements?
                            """).strip(),
                        },
                    ]
                )
            )

            # Strip images from old messages before trimming
            messages = self._strip_old_images(messages)
            state["messages"] = messages[-MAX_MESSAGES:]
            return state
        return state

    def analyze_with_labels(self, state: State) -> State:
        """Analyze using OmniParser structured data."""
        logger.info("Analyzing with parser...")
        messages = state.get("messages", [])
        parsed_content_list = state.get("parsed_content_list", [])
        labeled_img = state.get("labeled_img")

        # Always set parsed_content_list to an empty list if not found
        state["parsed_content_list"] = parsed_content_list or []

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
            logger.warning("No labeled image or parsed content list found")

        logger.info("Analysis with parser complete")
        # Strip images from old messages before trimming
        messages = self._strip_old_images(messages)
        state["messages"] = messages[-MAX_MESSAGES:]
        return state

    @log_runtime
    def decide_action(self, state: State, label_boxes: bool = True) -> State:
        """Decide action based on specified analysis mode."""
        logger.info("Deciding action...")

        if label_boxes:
            state = self.analyze_with_labels(state)
        else:
            state = self.analyze_raw_image(state)

        # Get response from LLM
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

                When clicking, you MUST use the box ID from the labeled image.
                The boxes are labeled in order of detection, starting from 0.
                Each box will show both its ID and content, like:
                - "0: File"
                - "1: What is the capital of France?"
                - "2: Paris"
                etc.

                Return a JSON object with:
                - "actions": A list of actions, each with:
                - "action": One of the available actions
                - "description": Why you chose this action
                - "parameters": Required parameters for the chosen action
                    - For 'click': Must include 'elementId' matching a box ID from the image
                    - For 'wait': Optional 'duration' in seconds (default: 2.0)
                    - For 'end': No parameters needed
            """).strip()
        )

        messages = [system_message] + state.get("messages", [])
        response = self.llm.invoke(messages)  # type: ignore
        if hasattr(response, "content"):
            response = ActionResponse(**json.loads(response.content))  # type: ignore
        assert isinstance(response, ActionResponse)
        state["actions"] = response.actions

        # Add AI message with complete context
        action_details = []
        for action in state["actions"]:
            params_str = ", ".join(f"{k}={v}" for k, v in action.parameters)
            logger.info(f"Decided action: {action.action} ({params_str}) - {action.description}")

            # Build detailed action message with reasoning
            action_msg = f"{action.action.upper()}"
            if action.parameters:
                param_details = [f"{k}={v}" for k, v in action.parameters]
                action_msg += f" ({', '.join(param_details)})"
            action_msg += f" - {action.description}"
            action_details.append(action_msg)

        # Add AI message with complete context
        message = "Actions decided:\n" + "\n→ ".join(action_details)
        state["messages"].append(AIMessage(content=message))

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
            action = action_obj.action
            if action == "end":
                logger.info("END action received - terminating")
                state["actions"] = []
                state["messages"].append(SystemMessage(content="Task ended."))
                return state

            parameters = action_obj.parameters
            logger.info(f"Executing action: {action} with parameters: {parameters}")

            # More detailed logging before execution
            if action == "click":
                if not isinstance(parameters, ClickParameters):
                    raise ValueError(f"Invalid parameters for click action: {parameters}")
                element_id = parameters.elementId

                if not element_id:
                    raise ValueError(f"Invalid element_id: {element_id}")

                label_coords = state["label_coordinates"]
                if label_coords is None:
                    raise ValueError("No label coordinates found in state")

                # Try exact match first
                coords = label_coords.get(element_id)

                # If not found and we're not using parser, try numeric IDs
                if coords is None and not self.label_boxes:
                    # Try to find a numeric ID that matches
                    for i in range(len(label_coords)):
                        if coords := label_coords.get(str(i)):
                            # Found a match, use these coordinates
                            logger.info(f"Using numeric ID {i} for click target")
                            break

                if coords is None:
                    raise ValueError(f"Element id {element_id} not found in label coordinates")

                rel_x = coords[0] + (coords[2] / 2)
                rel_y = coords[1] + (coords[3] / 2)
                abs_x, abs_y = self._convert_relative_to_absolute(rel_x, rel_y)

                if not self._click_at_coordinates(abs_x, abs_y, duration=0.3):
                    raise RuntimeError("Click action failed")
                time.sleep(0.5)
                logger.info(f"Clicked at ({abs_x}, {abs_y})")

            elif action == "wait":
                if isinstance(parameters, WaitParameters):
                    duration = parameters.duration
                else:
                    raise ValueError(f"Invalid parameters for wait action: {parameters}")
                time.sleep(duration)
                logger.info(f"Waited for {duration} seconds")

            elif action == "scroll":
                import pyautogui

                # Focus window first
                window_info = self.current_state["context"]["window_info"]
                if not window_info:
                    logger.error("No window info stored")
                    raise RuntimeError("Window info not found")

                # Simple keystroke command for each key
                key_commands = "\n".join([f'keystroke "{key}"' for key in ["down"]])

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
                time.sleep(0.1)

                # Scroll down
                if not isinstance(parameters, ScrollParameters):
                    # Convert empty dict to ScrollParameters with default amount
                    parameters = ScrollParameters()
                amount = parameters.amount
                pyautogui.scroll(amount)  # Negative values scroll down
                time.sleep(0.5)  # Wait for scroll to complete
                logger.info(f"Scrolled by {amount} units")

            elif action == "back":
                # Focus window first
                window_info = self.current_state["context"]["window_info"]
                if not window_info:
                    logger.error("No window info stored")
                    raise RuntimeError("Window info not found")

                # Send Cmd+Left Arrow to go back
                script = f"""
                    tell application "System Events"
                        tell process "{window_info['kCGWindowOwnerName']}"
                            set frontmost to true
                            key code 123 using command down  # Left arrow key
                        end tell
                    end tell
                """

                result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
                logger.info(f"Back navigation result: {result.stdout}")
                time.sleep(0.5)  # Wait for navigation to complete
                logger.info("Navigated back")

            else:
                raise ValueError(f"Unknown action: {action}")

            context["last_action"] = action

        # Clean up state after all actions complete
        state["actions"] = []
        state["context"] = context

        if not context.get("last_error"):
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

            return True

        except Exception as e:
            logger.error(f"Failed to click: {str(e)}")
            return False

    def _encode_image(self, image: Image.Image) -> str:
        """Convert image to base64 string with compression."""
        # Make a copy to avoid modifying original
        img = image.copy()

        # Resize if the image is too large
        max_size = 1024
        if img.width > max_size or img.height > max_size:
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Save as JPEG with higher compression
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85, optimize=True)
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

    def _strip_old_images(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Strip image content from all but the most recent message to save memory."""
        if not messages:
            return messages

        # Keep all messages but strip images from older ones
        result = []
        for i, msg in enumerate(messages):
            if i < len(messages) - 1 and isinstance(msg, HumanMessage):
                # For older messages, keep only the text content
                if isinstance(msg.content, list):
                    text_content = next((item["text"] for item in msg.content if item["type"] == "text"), None)  # type: ignore
                    if text_content:
                        result.append(HumanMessage(content=text_content))
                    else:
                        result.append(msg)  # Keep original if no text found
                else:
                    result.append(msg)  # Keep original if not multi-content
            else:
                result.append(msg)  # Keep the most recent message intact

        return result
