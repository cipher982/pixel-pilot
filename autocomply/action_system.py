import base64
from io import BytesIO
from typing import Annotated
from typing import Optional
from typing import TypedDict

import pyautogui
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from PIL import Image


class State(TypedDict):
    """State definition for the automation agent."""

    messages: Annotated[list, add_messages]  # Chat history
    screenshot: Optional[Image.Image]  # Current screen
    audio_text: Optional[str]  # Transcribed audio
    last_action: Optional[str]  # Track last action taken


class ActionSystem:
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-sonnet")
        self.graph = self._build_graph()
        pyautogui.FAILSAFE = True

    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def decide_action(self, state: State) -> dict:
        """Core decision making node that determines next action."""
        messages = [
            {
                "role": "system",
                "content": """You are an automation assistant that helps interact with educational content.
            Based on the screenshot and audio transcript, decide what action to take:
            1. WAIT - If content is still playing or no action needed
            2. CLICK x,y - To click a button or option at position x,y
            3. END - If the current section appears complete
            
            Respond in JSON format: {"action": "WAIT|CLICK|END", "coordinates": [x,y]}""",
            }
        ]

        if state["screenshot"]:
            base64_image = self._encode_image(state["screenshot"])
            messages.append({"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"})

        if state["audio_text"]:
            messages.append({"role": "user", "content": f"Audio transcript: {state['audio_text']}"})

        response = self.llm.invoke(messages)
        return {"decision": response.content}

    def execute_action(self, state: State) -> dict:
        """Execute the decided action."""
        decision = state.get("decision", {})
        if decision.get("action") == "CLICK":
            x, y = decision["coordinates"]
            pyautogui.moveTo(x, y)
            pyautogui.click()
            return {"last_action": f"Clicked at {x},{y}"}
        return {"last_action": "Waited"}

    def _build_graph(self) -> StateGraph:
        """Build the action graph."""
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("decide", self.decide_action)
        graph.add_node("execute", self.execute_action)

        # Add edges
        graph.add_edge("decide", "execute")
        graph.add_edge("execute", "decide")

        return graph.compile()

    def process(self, screenshot: Image.Image, audio_text: str) -> None:
        """Main entry point to process current state and take action."""
        state = {"messages": [], "screenshot": screenshot, "audio_text": audio_text, "last_action": None}

        for event in self.graph.stream(state):
            if "last_action" in event:
                print(f"Action taken: {event['last_action']}")
