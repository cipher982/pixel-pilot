import base64
import json
import time
from io import BytesIO
from textwrap import dedent
from typing import Annotated
from typing import Optional
from typing import TypedDict

import pyautogui
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
    def __init__(self):
        self.llm = ChatOpenAI(model=MODEL_NAME)
        pyautogui.FAILSAFE = True
        # Store state between runs
        self.current_state = {
            "messages": [],
            "context": {
                "last_action": None,
                "waiting_for_audio": False,
                "tab_count": 0,  # Track tab position
            },
            "parameters": {},
        }
        self.graph = self._build_graph()
        logger.info("Action system initialized")

    def should_decide(self, state: State) -> str:
        """Determine if we should move to decide step."""
        logger.info(f"should_decide: action={state.get('action')}")
        return "decide" if state.get("action") == "ANALYZED" else "analyze"

    def should_execute(self, state: State) -> str:
        """Determine if we should move to execute step."""
        logger.info(f"should_execute: action={state.get('action')}")
        return "execute" if state.get("action") in ["CLICK", "TYPE", "WAIT", "END"] else "analyze"

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
            # Process screenshot and build message
            base64_image = self._encode_image(state["screenshot"])
            messages.append(
                HumanMessage(
                    content=[
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {"type": "text", "text": "What has changed in this screenshot? What action should we take?"},
                    ],
                )
            )

            return {"messages": messages, "action": "ANALYZED", "parameters": {}, "context": state["context"]}

        return {"messages": messages, "action": "WAIT", "parameters": {}, "context": state["context"]}

    def decide_action(self, state: State) -> dict:
        """Decide what action to take based on analysis."""
        system_message = SystemMessage(
            content=dedent("""
                You are an automation assistant analyzing a sequence of screenshots.
                The content supports keyboard navigation with the following actions:
                - TAB: Move between interactive elements
                - ENTER: Activate current element
                - SHIFT_ALT_PLUS: Advance to next slide
                - WAIT: Wait for audio or animations
                
                If you see:
                - A new slide: Start with TAB to begin navigation
                - A question or interactive element: Use TAB until focused, then ENTER
                - Content finished playing: Try SHIFT_ALT_PLUS to advance
                - Loading or playing audio: Use WAIT
                
                Return a JSON object with:
                - "action": One of TAB, ENTER, SHIFT_ALT_PLUS, WAIT
                - "parameters": Additional context (optional)
                - "description": What you see and why you chose this action
            """).strip()
        )

        messages = [system_message] + state.get("messages", [])

        try:
            response = self.llm.invoke(messages, response_format={"type": "json_object"}).content
            response = json.loads(response)
            logger.info(f"Decision: {response}")
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
        parameters = state.get("parameters", {})
        context = state.get("context", {})
        logger.info(f"Executing action: {action} with parameters: {parameters}")

        try:
            if action == "WAIT":
                time.sleep(2)  # Adjustable wait time
                return {"messages": ["Waited"], "action": None, "context": context}

            elif action == "TAB":
                pyautogui.press("tab")
                context["tab_count"] = context.get("tab_count", 0) + 1
                time.sleep(0.5)

            elif action == "ENTER":
                pyautogui.press("enter")
                context["tab_count"] = 0  # Reset tab count after interaction
                time.sleep(0.5)

            elif action == "SHIFT_ALT_PLUS":
                pyautogui.hotkey("shift", "alt", "+")
                context["tab_count"] = 0  # Reset tab count on new slide
                time.sleep(1)

            context["last_action"] = action
            return {"messages": [f"Successfully executed {action}"], "action": None, "context": context}

        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            return {"messages": [f"Failed to execute {action}"], "action": "WAIT"}

    def _build_graph(self) -> StateGraph:
        """Build the action graph with analysis, decision, and execution nodes."""
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("analyze", self.analyze_inputs)
        workflow.add_node("decide", self.decide_action)
        workflow.add_node("execute", self.execute_action)

        # Set entry point explicitly
        workflow.set_entry_point("analyze")

        # Add conditional edges with proper routing
        workflow.add_conditional_edges("analyze", self.should_decide, {"decide": "decide", "analyze": "analyze"})

        workflow.add_conditional_edges("decide", self.should_execute, {"execute": "execute", "analyze": "analyze"})

        # Add direct edge back to analyze after execution
        workflow.add_edge("execute", "analyze")

        return workflow.compile()

    def run(self, screenshot: Optional[Image.Image], audio_text: Optional[str]) -> list:
        """Run one iteration of the graph with current inputs."""
        initial_state = {
            "messages": self.current_state["messages"],
            "screenshot": screenshot,
            "audio_text": audio_text,
            "action": None,
            "context": self.current_state["context"],
            "parameters": self.current_state["parameters"],
        }

        logger.info("Running action graph...")
        events = list(self.graph.stream(initial_state))
        logger.info(f"Graph generated {len(events)} events")
        return events
