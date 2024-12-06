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
    client_config: InferenceClientConfig = Field(description="Configuration for the TGI client")

    def __init__(self, base_url: str, timeout: float = 60.0, **kwargs: Any):
        client_config = InferenceClientConfig(base_url=base_url, timeout=timeout)
        self.client_config = client_config
        super().__init__(**kwargs)

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
        formatted_messages = [{"role": msg.type, "content": str(msg.content)} for msg in messages]

        # Make the chat completion request
        response = client.chat_completion(messages=formatted_messages, **kwargs)

        # Extract content from response
        content = response.choices[0].message.content
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
