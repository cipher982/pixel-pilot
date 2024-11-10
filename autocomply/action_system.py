import base64
from io import BytesIO
from typing import Annotated
from typing import Optional
from typing import TypedDict

import pyautogui
from langgraph.graph import END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from PIL import Image

from autocomply.logger import setup_logger

logger = setup_logger(__name__)


class State(TypedDict):
    """State definition for the automation agent."""

    messages: Annotated[list, add_messages]
    screenshot: Optional[Image.Image]
    audio_text: Optional[str]
    action: Optional[str]


class ActionSystem:
    def __init__(self):
        self.llm = wrap_openai(OpenAI())
        pyautogui.FAILSAFE = True
        self.graph = self._build_graph()

    def _encode_image(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def decide_action(self, state: State) -> dict:
        """Decide what action to take based on current state."""
        logger.info("Deciding next action...")

        messages = [
            {
                "role": "system",
                "content": """
                You are an automation assistant. 
                Decide what action to take based on the screenshot and audio.
                """,
            }
        ]

        if state["screenshot"]:
            base64_image = self._encode_image(state["screenshot"])
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image_url": f"data:image/png;base64,{base64_image}"},
                        {"type": "text", "text": "Here is the current screen state."},
                    ],
                }
            )

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini", messages=messages, response_format={"type": "json_object"}
            )
            decision = response.choices[0].message.content
            logger.info(f"Decision: {decision}")
            return {"messages": [decision], "action": decision["action"]}
        except Exception as e:
            logger.error(f"Error in decide_action: {e}")
            return {"messages": [], "action": "WAIT"}

    def execute_action(self, state: State) -> dict:
        """Execute the decided action."""
        action = state["action"]
        logger.info(f"Executing action: {action}")

        if action == "WAIT":
            return {"messages": ["Waited"], "action": None}
        elif action == "END":
            return END

        # Execute other actions here
        return {"messages": [f"Executed {action}"], "action": None}

    def _build_graph(self) -> StateGraph:
        """Build the action graph."""
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("decide", self.decide_action)
        graph.add_node("execute", self.execute_action)

        # Add edges
        graph.add_edge("decide", "execute")
        graph.add_edge("execute", "decide")

        logger.info("Action graph built successfully")
        return graph.compile()

    def run(self, screenshot: Optional[Image.Image], audio_text: Optional[str]) -> list:
        """Run one iteration of the graph with current inputs."""
        initial_state = {
            "messages": [],
            "screenshot": screenshot,
            "audio_text": audio_text,
            "action": None,
        }

        logger.info("Running action graph...")
        events = list(self.graph.stream(initial_state))
        logger.info(f"Graph generated {len(events)} events")
        return events
