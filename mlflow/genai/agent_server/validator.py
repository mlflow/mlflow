import logging
from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel

from mlflow.genai.agent_server.types import AgentType
from mlflow.types.agent import ChatAgentChunk, ChatAgentRequest, ChatAgentResponse
from mlflow.types.llm import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.utils.annotations import experimental


@experimental(version="3.6.0")
class AgentValidator:
    def __init__(self, agent_type: AgentType | None = None):
        self.agent_type = agent_type
        self.logger = logging.getLogger(__name__)

    def validate_pydantic(self, pydantic_class: type[BaseModel], data: Any) -> None:
        """Generic pydantic validator that throws an error if the data is invalid"""
        if isinstance(data, pydantic_class):
            return
        try:
            if isinstance(data, BaseModel):
                pydantic_class(**data.model_dump())
                return
            pydantic_class(**data)
        except Exception as e:
            raise ValueError(
                f"Invalid data for {pydantic_class.__name__} (agent_type: {self.agent_type}): {e}"
            )

    def validate_dataclass(self, dataclass_class: Any, data: Any) -> None:
        """Generic dataclass validator that throws an error if the data is invalid"""
        if isinstance(data, dataclass_class):
            return
        try:
            dataclass_class(**data)
        except Exception as e:
            raise ValueError(
                f"Invalid data for {dataclass_class.__name__} (agent_type: {self.agent_type}): {e}"
            )

    def validate_and_convert_request(
        self, data: dict[str, Any]
    ) -> dict[str, Any] | ResponsesAgentRequest | ChatCompletionRequest | ChatAgentRequest:
        """Validate request parameters based on agent type"""
        if self.agent_type is None:
            return data
        elif self.agent_type == "agent/v1/responses":
            self.validate_pydantic(ResponsesAgentRequest, data)
            return ResponsesAgentRequest(**data)
        elif self.agent_type == "agent/v1/chat":
            for msg in data.get("messages", []):
                self.validate_dataclass(ChatMessage, msg)
            return ChatCompletionRequest(**data)
        elif self.agent_type == "agent/v2/chat":
            self.validate_pydantic(ChatAgentRequest, data)
            return ChatAgentRequest(**data)

    def validate_invoke_response(self, result: Any) -> None:
        if self.agent_type == "agent/v1/responses":
            self.validate_pydantic(ResponsesAgentResponse, result)
        elif self.agent_type == "agent/v1/chat":
            self.validate_dataclass(ChatCompletionResponse, result)
        elif self.agent_type == "agent/v2/chat":
            self.validate_pydantic(ChatAgentResponse, result)

    def validate_stream_response(self, result: Any) -> None:
        if self.agent_type == "agent/v1/responses":
            self.validate_pydantic(ResponsesAgentStreamEvent, result)
        elif self.agent_type == "agent/v1/chat":
            self.validate_dataclass(ChatCompletionChunk, result)
        elif self.agent_type == "agent/v2/chat":
            self.validate_dataclass(ChatAgentChunk, result)

    def validate_and_convert_result(self, result: Any, stream: bool = False) -> dict[str, Any]:
        """Validate and convert the result into a dictionary if necessary"""
        if stream:
            self.validate_stream_response(result)
        else:
            self.validate_invoke_response(result)

        if isinstance(result, BaseModel):
            return result.model_dump(exclude_none=True)
        elif is_dataclass(result):
            return asdict(result)
        elif isinstance(result, dict):
            return result
        else:
            raise ValueError(
                f"Result needs to be a pydantic model, dataclass, or dict. "
                f"Unsupported result type: {type(result)}, result: {result}"
            )
