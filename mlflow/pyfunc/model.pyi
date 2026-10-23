from collections.abc import Generator
from typing import Any

from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.types.llm import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
    ChatParams,
)

class PythonModelContext:
    def __init__(self, artifacts: dict[str, str], model_config: dict[str, Any]) -> None: ...
    @property
    def artifacts(self) -> dict[str, str]: ...
    @property
    def model_config(self) -> dict[str, Any]: ...

class PythonModel:
    def load_context(self, context: PythonModelContext) -> None: ...
    def predict(
        self,
        model_input: Any,
        params: dict[str, Any] | None = ...,
    ) -> Any: ...
    def predict_stream(
        self,
        model_input: Any,
        params: dict[str, Any] | None = ...,
    ) -> Generator[Any, None, None]: ...

class ChatModel(PythonModel):
    def predict(
        self,
        messages: list[ChatMessage],
        params: ChatParams,
    ) -> ChatCompletionResponse: ...
    def predict_stream(
        self,
        messages: list[ChatMessage],
        params: ChatParams,
    ) -> Generator[ChatCompletionChunk, None, None]: ...

class ChatAgent(PythonModel):
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: ChatContext | None = ...,
        custom_inputs: dict[str, Any] | None = ...,
    ) -> ChatAgentResponse: ...
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: ChatContext | None = ...,
        custom_inputs: dict[str, Any] | None = ...,
    ) -> Generator[ChatAgentChunk, None, None]: ...

class ResponsesAgent(PythonModel):
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse: ...
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]: ...
