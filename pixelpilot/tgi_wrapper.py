import json
from typing import Any
from typing import List
from typing import Optional

from huggingface_hub import InferenceClient
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult
from pydantic import BaseModel
from pydantic import Field


class InferenceClientConfig(BaseModel):
    base_url: str = Field(description="Base URL for the TGI server")
    timeout: float = Field(default=60.0, description="Timeout for requests in seconds")


class LocalTGIChatModel(BaseChatModel):
    client_config: InferenceClientConfig
    response_schema: Optional[type[BaseModel]] = None

    def __init__(self, base_url: str, timeout: float = 60.0, **kwargs: Any):
        client_config = InferenceClientConfig(base_url=base_url, timeout=timeout)
        kwargs["client_config"] = client_config
        super().__init__(**kwargs)

    def with_structured_output(self, schema: type[BaseModel]) -> "LocalTGIChatModel":
        """Configure the model to return structured output according to the given schema."""
        self.response_schema = schema
        return self

    @property
    def _llm_type(self) -> str:
        return "local-tgi-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not messages:
            raise ValueError("At least one HumanMessage must be provided!")

        client = InferenceClient(self.client_config.base_url)

        # Format messages for the inference client
        formatted_messages = []
        for msg in messages:
            if isinstance(msg.content, list):
                # Handle multimodal content
                formatted_messages.append(
                    {
                        "role": msg.type,
                        "content": msg.content,  # Keep structured content as-is
                    }
                )
            else:
                # Handle text-only content
                formatted_messages.append({"role": msg.type, "content": str(msg.content)})

        # Add response format if schema is set
        kwargs_with_format = kwargs.copy()
        if self.response_schema:
            kwargs_with_format["response_format"] = {
                "type": "json_object",
                "value": self.response_schema.model_json_schema(),
            }

        # Make the chat completion request
        response = client.chat.completions.create(messages=formatted_messages, **kwargs_with_format)

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

        # Return a ChatResult
        return ChatResult(
            generations=[gen],
            llm_output={
                "model": self.client_config.base_url,
                "token_usage": {
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            },
        )
