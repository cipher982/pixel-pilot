import base64
import json
import os
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
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
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

from pixelpilot.action_utils import back_action
from pixelpilot.action_utils import click_at_coordinates
from pixelpilot.action_utils import convert_relative_to_absolute
from pixelpilot.action_utils import scroll_action
from pixelpilot.audio_capture import AudioCapture
from pixelpilot.logger import setup_logger
from pixelpilot.openai_wrapper import OpenAICompatibleChatModel
from pixelpilot.tgi_wrapper import LocalTGIChatModel
from pixelpilot.utils import get_som_labeled_img
from pixelpilot.utils import log_runtime
from pixelpilot.window_capture import WindowCapture

# Logger
logger = setup_logger(__name__)

# Semantic models
CAPTION_MODEL = "florence"
# CAPTION_MODEL = "blip"
# ENABLE_FLORENCE_CAPABILITY = False
# DECISION_MODEL = "gpt-4o-2024-08-06"
MAX_MESSAGES = 10

# LLM Models
OPENAI_MODEL = "gpt-4o"
# BEDROCK_MODEL = "us.amazon.nova-lite-v1:0"
# BEDROCK_MODEL = "us.amazon.nova-pro-v1:0"
# BEDROCK_MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
BEDROCK_MODEL = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
FIREWORKS_MODEL = "accounts/fireworks/models/phi-3-vision-128k-instruct"
LOCAL_MODEL_URL = "http://jelly:8080"


# Constants
WAIT_DURATION = 2.0  # Default wait duration in seconds
SCROLL_AMOUNT = -3400  # Default scroll amount in pixels


class ClickAction(BaseModel):
    """Action to click an element."""

    reason: str = Field(description="Why this element should be clicked, and whats the answer.")
    action_type: Literal["click"] = Field(description="Click action")
    element_id: str = Field(description="ID of the element to click")


class ScrollAction(BaseModel):
    """Action to scroll the page."""

    reason: str = Field(description="Why we need to scroll")
    action_type: Literal["scroll"] = Field(description="Scroll action")


class WaitAction(BaseModel):
    """Action to wait."""

    reason: str = Field(description="Why this action was chosen")
    action_type: Literal["wait"] = Field(description="Wait action")


class EndAction(BaseModel):
    """Action to end."""

    reason: str = Field(description="Why this action was chosen")
    action_type: Literal["end"] = Field(description="End action")


class BackAction(BaseModel):
    """Action to go back."""

    reason: str = Field(description="Why this action was chosen")
    action_type: Literal["back"] = Field(description="Back action")


class TerminalAction(BaseModel):
    """Action to execute terminal commands."""

    reason: str = Field(description="Why this command needs to be executed")
    action_type: Literal["terminal"] = Field(description="Terminal action")
    command: str = Field(description="The command to execute")


class ActionUnion(BaseModel):
    """Union of all possible actions. Include "action" key."""

    action: Union[ClickAction, ScrollAction, WaitAction, EndAction, BackAction, TerminalAction] = Field(
        description="Action to take"
    )


class State(TypedDict):
    """Define the state schema and reducers."""

    messages: Annotated[list, add_messages]
    screenshot: Optional[Image.Image]
    audio_text: Optional[str]
    actions: List[ActionUnion]
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
        if llm_provider == "local":
            self.llm = LocalTGIChatModel(base_url=LOCAL_MODEL_URL).with_structured_output(ActionUnion)
        elif llm_provider == "openai":
            self.llm = ChatOpenAI(model=OPENAI_MODEL).with_structured_output(ActionUnion)
        elif llm_provider == "fireworks":
            self.llm = OpenAICompatibleChatModel(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.getenv("FIREWORKS_API_KEY"),  # type: ignore
                model=FIREWORKS_MODEL,
            ).with_structured_output(ActionUnion)
        elif llm_provider == "bedrock":
            # Use Bedrock with structured output
            self.llm = ChatBedrockConverse(
                credentials_profile_name="preprod",
                region_name="us-east-1",
                model=BEDROCK_MODEL,
                temperature=0,
                max_tokens=None,
            ).with_structured_output(ActionUnion)
            logger.info(f"Initialized Bedrock LLM with model: {BEDROCK_MODEL}")
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
        workflow.add_node("analyze_raw_image", self.analyze_raw_image)  # Add analyze_raw_image node
        workflow.add_node("decide_action", self.decide_action)  # type: ignore
        workflow.add_node("execute_action", self.execute_action)

        # Set entry point
        workflow.set_entry_point("capture_state")

        # Add edges with END check
        workflow.add_edge("capture_state", "analyze_raw_image")  # Go to analyze first
        workflow.add_edge("analyze_raw_image", "decide_action")  # Then to decide
        workflow.add_conditional_edges(
            "decide_action",
            lambda state: "end" if any(a.action.action_type == "end" for a in state["actions"]) else "execute",
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

            # # Save image to temp file
            # img.save("screenshot.png")

            timestamp = time.strftime("%H:%M:%S")

            # Create a list of available box IDs
            assert label_coordinates, "label_coordinates should not be empty"
            available_boxes = sorted(label_coordinates.keys())
            box_info = ", ".join([f"{box_id}" for box_id in available_boxes])

            messages.append(
                HumanMessage(
                    content=[
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{compressed_base64}"}},
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

    @log_runtime
    def decide_action(self, state: State) -> State:
        """Decide what action to take based on the current state."""
        if not state["messages"]:
            raise ValueError("No messages in state")

        if state["screenshot"] is None:
            raise ValueError("No screenshot in state")

        system_message = SystemMessage(
            content=dedent(f"""
                    You are an automation assistant analyzing screenshots.
                    
                    Task Instructions:
                    {self.config["instructions"]}

                    When clicking, you MUST use the box ID from the labeled image.
                    The boxes are labeled in order of detection, starting from 0.
                    Each box will show both its ID and contenst, like:
                    - "0: File"
                    - "1: What is the capital of France?"
                    - "2: Paris"
                    etc.

                    Your response MUST be in this format:
                    (Dont forget the first key as "action"!):

                    For clicking:
                    {{"action": {{"action_type": "click", "reason": "why clicking", "element_id": "box_id"}}}}

                    For scrolling:
                    {{"action": {{"action_type": "scroll", "reason": "why scrolling", "amount": scroll_amount}}}}

                    For wait/end/back:
                    {{"action": {{"action_type": "wait|end|back", "reason": "why this action"}}}}

                    These should be output as tool calls, not standard message content!!
                """).strip()
        )

        # Existing logic for capturing state and preparing messages
        state = self.analyze_raw_image(state)
        messages = [system_message] + state.get("messages", [])

        response = self.llm.invoke(messages)  # type: ignore
        logger.debug(f"Raw LLM Response: {response}")
        logger.debug(f"Response Type: {type(response)}")
        print(f"Parsed LLM Response: {response}")
        print(f"Response Type: {type(response)}")

        # Parse the response content into ActionUnion
        if isinstance(response, AIMessage):
            content = json.loads(response.content)  # type: ignore
            action = ActionUnion.model_validate(content)
        else:
            action: ActionUnion = response  # type: ignore

        state["actions"] = [action]
        state["messages"].append(AIMessage(content=json.dumps(action.model_dump())))  # type: ignore

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

            # WAIT
            if isinstance(action, WaitAction):
                time.sleep(3.0)
                logger.info(f"Waited for {3.0} seconds")
                state["actions"] = []
                return state

            # END
            elif isinstance(action, EndAction):
                logger.info(f"END action received - terminating. Reason: {action.reason}")
                state["actions"] = []
                state["messages"].append(SystemMessage(content="Task ended."))
                return state

            # BACK
            elif isinstance(action, BackAction):
                logger.info(f"BACK action received. Reason: {action.reason}")
                back_action(self.current_state["context"]["window_info"])
                state["actions"] = []
                return state

            # CLICK
            elif isinstance(action, ClickAction):
                print("Click action received")
                print(f"element to click: {action.element_id}")
                print(f"reason: {action.reason}")

                element_id = action.element_id
                if not element_id:
                    raise ValueError("Missing required elementId for click action")

                label_coords = state["label_coordinates"]
                if label_coords is None:
                    raise ValueError("No label coordinates found in state")

                coords = label_coords.get(element_id)
                if coords is None:
                    raise ValueError(f"Element id {element_id} not found in label coordinates")

                abs_x, abs_y = convert_relative_to_absolute(
                    window_info=self.current_state["context"]["window_info"],
                    rel_x=coords[0] + (coords[2] / 2),
                    rel_y=coords[1] + (coords[3] / 2),
                )
                if not click_at_coordinates(
                    window_info=self.current_state["context"]["window_info"],
                    x=abs_x,
                    y=abs_y,
                    duration=0.3,
                ):
                    raise RuntimeError("Click action failed")

                time.sleep(0.5)
                logger.info(f"Clicked at ({abs_x}, {abs_y})")
            # SCROLL
            elif isinstance(action, ScrollAction):
                print("Scroll action received")
                print(f"reason: {action.reason}")
                scroll_action(self.current_state["context"]["window_info"])
            elif isinstance(action, TerminalAction):
                print("Terminal action received")
                print(f"command: {action.command}")
                print(f"reason: {action.reason}")

                import subprocess

                try:
                    result = subprocess.run(action.command, shell=True, check=True, capture_output=True, text=True)
                    logger.info(f"Command executed successfully: {result.stdout}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Command failed: {e.stderr}")
                    raise RuntimeError(f"Terminal command failed: {e}")

                time.sleep(0.5)  # Small delay after command execution
            else:
                raise ValueError(f"Unknown action: {action}")

            context["last_action"] = action

        # Clean up state after all actions complete
        state["actions"] = []
        state["context"] = context

        if not context.get("last_error"):
            logger.info("All actions completed successfully")

        return state

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

        # Save as PNG
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

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
