import json
from typing import Any
from typing import Dict
from typing import List

from openai import OpenAI


class TGIClient:
    """Simple wrapper for OpenAI client configured to use TGI."""

    def __init__(
        self,
        base_url: str = "http://jelly:8080/v1",
        api_key: str = "-",
        model: str = "tgi",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def _convert_message(self, message: Any) -> Dict[str, str]:
        """Convert a LangChain message to OpenAI format."""
        if hasattr(message, "content") and hasattr(message, "type"):
            # Convert LangChain message types
            role = "system" if message.type == "system" else "user" if message.type == "human" else "assistant"

            # Handle multimodal content
            content = message.content
            if isinstance(content, list):
                # Extract only text content from multimodal messages
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part["text"])
                content = "\n".join(text_parts)

            return {"role": role, "content": content}
        elif isinstance(message, dict) and "content" in message:
            # Already in OpenAI format or close to it
            return {"role": message.get("role", "user"), "content": message["content"]}
        else:
            raise ValueError(f"Unsupported message format: {message}")

    def invoke(self, messages: List[Any], **kwargs: Any):
        """Create a chat completion using TGI and parse the response."""
        openai_messages = [self._convert_message(msg) for msg in messages]
        response = self.client.chat.completions.create(model=self.model, messages=openai_messages, **kwargs)
        # Parse the JSON response into an ActionResponse
        try:
            content = response.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")
