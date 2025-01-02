import json
from typing import Any
from typing import List
from typing import Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult
from openai import OpenAI
from pydantic import BaseModel
from pydantic import Field


class OpenAIClientConfig(BaseModel):
    """Configuration for OpenAI-compatible client."""

    base_url: str = Field(description="Base URL for the OpenAI-compatible API")
    api_key: str = Field(description="API key for authentication")
    model: str = Field(description="Model identifier to use")
    timeout: float = Field(default=60.0, description="Timeout for requests in seconds")


class OpenAICompatibleChatModel(BaseChatModel):
    """Chat model for OpenAI-compatible APIs with JSON mode support."""

    client_config: OpenAIClientConfig
    response_schema: Optional[type[BaseModel]] = None

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0, **kwargs: Any):
        client_config = OpenAIClientConfig(base_url=base_url, api_key=api_key, model=model, timeout=timeout)
        kwargs["client_config"] = client_config
        super().__init__(**kwargs)

    def with_structured_output(self, schema: type[BaseModel]) -> "OpenAICompatibleChatModel":
        """Configure the model to return structured output according to the given schema."""
        self.response_schema = schema
        return self

    @property
    def _llm_type(self) -> str:
        return "openai-compatible-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not messages:
            raise ValueError("At least one message must be provided!")

        client = OpenAI(
            base_url=self.client_config.base_url, api_key=self.client_config.api_key, timeout=self.client_config.timeout
        )

        print(f"CLIENT CONFIG: {self.client_config}")

        # Format messages for the OpenAI client
        formatted_messages = []
        for msg in messages:
            if isinstance(msg.content, list):
                # For multimodal content, pass the content array directly
                formatted_messages.append(
                    {
                        "role": msg.type,
                        "content": msg.content,  # Pass through as-is
                    }
                )
            else:
                # For text-only content
                formatted_messages.append({"role": msg.type, "content": [{"type": "text", "text": str(msg.content)}]})

        print("CLIENT CONFIG:", self.client_config)

        # Add response format if schema is set
        kwargs_with_format = kwargs.copy()
        if self.response_schema:
            kwargs_with_format["response_format"] = {
                "type": "json_object",
                "schema": self.response_schema.model_json_schema(),
            }

        # print("DEBUG FORMATTED MESSAGES:", formatted_messages)

        # Make the chat completion request
        response = client.chat.completions.create(
            model=self.client_config.model, messages=formatted_messages, **kwargs_with_format
        )

        # Extract content from response
        content = response.choices[0].message.content

        # Parse JSON response if schema is set
        if self.response_schema:
            try:
                content = json.loads(content)
                content = self.response_schema.model_validate(content)
                # Convert Pydantic model to dict for AIMessage
                content = content.model_dump()
                # Convert dict back to JSON string
                content = json.dumps(content)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Failed to parse response as {self.response_schema.__name__}: {e}")

        finish_reason = response.choices[0].finish_reason

        # Create a single ChatGeneration with the response
        gen = ChatGeneration(
            message=AIMessage(content=content),
            generation_info={
                "finish_reason": finish_reason,
                "usage": {
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            },
        )

        return ChatResult(generations=[gen])
