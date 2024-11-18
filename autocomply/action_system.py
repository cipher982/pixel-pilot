import base64
import subprocess
import time
from io import BytesIO
from textwrap import dedent
from typing import Annotated
from typing import Literal
from typing import Optional
from typing import Type
from typing import TypedDict
from typing import Union

import numpy as np
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
from pydantic import create_model

from autocomply.audio_capture import AudioCapture
from autocomply.logger import setup_logger
from autocomply.window_capture import WindowCapture

logger = setup_logger(__name__)

MODEL_NAME = "gpt-4o-2024-08-06"
MAX_MESSAGES = 5
# USE_PARSER = False


class State(TypedDict):
    """Define the state schema and reducers."""

    messages: Annotated[list, add_messages]
    screenshot: Optional[Image.Image]
    audio_text: Optional[str]
    action: Optional[str]
    parameters: dict
    context: dict
    labeled_img: Optional[str]  # base64 encoded image
    label_coordinates: Optional[dict]
    parsed_content_list: Optional[list]


class ActionSystem:
    def __init__(
        self,
        task_profile: Optional[str] = None,
        instructions: Optional[str] = None,
        no_audio: bool = False,
        debug: bool = False,
        use_parser: bool = False,
    ):
        self.config = self._load_task_config(task_profile, instructions)
        self.response_model = self._create_pydantic_model(self.config)
        self.llm = ChatOpenAI(model=MODEL_NAME).with_structured_output(self.response_model)
        self.debug = debug
        self.use_parser = use_parser

        # Initialize placeholders for all models
        self._yolo_model = None
        self._florence_processor = None
        self._florence_model = None

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
            )
            logger.info("Florence caption model loaded successfully")
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

    def _create_pydantic_model(self, config: dict) -> Type[BaseModel]:
        """Convert YAML schema to Pydantic model dynamically"""
        schema = config["schema"]

        # Create parameter models from oneOf schemas
        parameter_schemas = schema["properties"]["parameters"]["oneOf"]
        parameter_models = []

        for param_schema in parameter_schemas:
            fields = {}
            # Add fields only if properties exist
            if "properties" in param_schema:
                for field_name, field_info in param_schema["properties"].items():
                    field_type = str if field_info["type"] == "string" else float
                    # Make all fields optional with None as default
                    fields[field_name] = (Optional[field_type], None)

            # Create model even if no fields (for empty parameters)
            param_model = create_model(f"Parameters_{len(parameter_models)}", **fields)
            parameter_models.append(param_model)

        # Build main model fields
        fields = {
            "action": (Literal.__getitem__(tuple(schema["properties"]["action"]["enum"])), ...),
            "description": (str, ...),
            "parameters": (Union.__getitem__(tuple(parameter_models)), {}),
        }
        pydantic_model = create_model("ActionResponse", **fields)
        logger.info(f"Pydantic model created: {pydantic_model}")
        return pydantic_model

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
            lambda state: "end" if state["action"] == "END" else "execute",
            {"end": END, "execute": "execute_action"},
        )
        workflow.add_edge("execute_action", "capture_state")

        return workflow.compile()

    # def should_continue(self, state: State) -> str:
    #     """Determine whether to proceed to execute_action or end."""
    #     action = state.get("action")
    #     if action == "END":
    #         return "end"
    #     else:
    #         return "execute_action"

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

    def capture_state(self, state: State) -> State:
        """Capture the current state."""
        logger.info("Capturing state...")

        # Capture screenshot
        screenshot = self.window_capture.capture_window(self.window_info)
        state["screenshot"] = screenshot

        if self.use_parser:
            from autocomply.utils import check_ocr_box
            from autocomply.utils import get_som_labeled_img

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
            compressed_base64 = self._encode_image(img, max_size=1000)

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
            response = self.llm.invoke(messages)
            action = response.action.lower()

            # Early return for END action
            if action == "end":
                logger.info("END action received - terminating")
                state["action"] = "END"
                state["parameters"] = {}
                state["messages"].append(AIMessage(content=f"Decision: {response.description} -> Action: END"))
                return state

            # Continue processing for other actions
            state["action"] = action
            if response.parameters and hasattr(response.parameters, "model_dump"):
                params = response.parameters.model_dump(exclude_none=True, exclude_unset=True)
                state["parameters"] = params if params else {}
            else:
                state["parameters"] = {}

            state["messages"].append(
                AIMessage(content=f"Decision: {response.description} -> Action: {state['action']}")
            )

            logger.info(f"Decided action: {state['action']} with parameters: {state['parameters']}")
            return state

        except Exception as e:
            logger.error(f"Action decision failed: {str(e)}")
            state["action"] = "END"  # Fail safe to END
            return state

    def execute_action(self, state: State) -> State:
        """Execute the decided action with improved handling and feedback."""
        logger.info("Executing action...")
        action = state.get("action", "").lower()

        if action == "end":
            logger.info("END action received - terminating")
            state["action"] = "END"
            return state

        parameters = state.get("parameters", {})
        context = state.get("context", {})

        logger.info(f"Executing action: {action} with parameters: {parameters}")

        try:
            match action:
                case "click":
                    element_id = parameters.get("elementId") or parameters.get("element_id")
                    if element_id is not None:
                        element_id = str(element_id)
                        if element_id not in state["label_coordinates"]:
                            raise ValueError(f"Invalid element_id: {element_id}")
                    else:
                        raise ValueError("No element_id provided in parameters")
                    logger.info(f"Using element_id: {element_id}")

                    rel_coords = state["label_coordinates"][element_id]
                    logger.info(f"Found relative coordinates: {rel_coords}")

                    # Calculate center point
                    rel_x = rel_coords[0] + (rel_coords[2] / 2)
                    rel_y = rel_coords[1] + (rel_coords[3] / 2)
                    abs_x, abs_y = self._convert_relative_to_absolute(rel_x, rel_y)
                    logger.info(f"Using absolute coordinates: ({abs_x}, {abs_y})")

                    # Pre-click delay for visibility
                    time.sleep(0.5)
                    success = self._click_at_coordinates(abs_x, abs_y)
                    logger.info(f"Click action result: {success}")

                    if not success:
                        raise RuntimeError("Click action failed")

                    # Post-click delay
                    time.sleep(0.5)

                case "wait":
                    logger.info("Waiting for content...")
                    time.sleep(parameters.get("duration", 2.0))

                case "end":
                    logger.info("Task completed")
                    state["action"] = "END"
                    return state

                case "type" if "text" in parameters:
                    # Optional: Implement text input if needed
                    text = parameters["text"]
                    logger.info(f"Typing text: {text}")
                    # TODO: Implement typing mechanism
                    pass

                case _:
                    raise ValueError(f"Unknown action: {action}")

            # Update context with successful action
            context["last_action"] = action
            state["context"] = context

            # Only reset action for non-end actions
            if action != "end":
                state["action"] = None

            logger.info("Action executed successfully")
            return state

        except Exception as e:
            logger.error(f"Action execution failed: {str(e)}")
            # Update state to reflect failure
            state["action"] = None
            state["context"] = {**context, "last_error": str(e), "last_failed_action": action}
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

    def _click_at_coordinates(self, x: int, y: int, duration: float = 1.0) -> bool:
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
            time.sleep(0.5)

            # Now perform the click
            pyautogui.moveTo(x, y, duration=duration)
            pyautogui.click()

            logger.info(f"Clicked at ({x}, {y})")
            return True

        except Exception as e:
            logger.error(f"Failed to click: {str(e)}")
            return False

    def _encode_image(self, image: Image.Image, max_size: int = 800) -> str:
        """Encode screenshot to base64 string with PNG optimization."""
        img = image.copy()

        # Resize if larger than max_size
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to PNG with optimization
        buffered = BytesIO()
        img.save(buffered, format="PNG", optimize=True)
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
        from autocomply.utils import check_ocr_box
        from autocomply.utils import get_som_labeled_img

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
