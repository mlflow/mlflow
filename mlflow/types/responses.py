from typing import Any, Optional, Union

from mlflow.types.chat import BaseModel
from mlflow.types.responses_helpers import (
    BaseRequestPayload,
    ComputerCallOutput,
    ComputerTool,
    EasyInputMessageParam,
    FileSearchTool,
    FunctionCallOutput,
    FunctionTool,
    ItemReference,
    Message,
    Response,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
    ResponseReasoningItem,
    WebSearchTool,
)
from mlflow.types.type_hints import _infer_schema_from_type_hint


class ResponsesRequest(BaseRequestPayload):
    input: list[
        Union[
            EasyInputMessageParam,
            Message,
            ResponseOutputMessage,
            ResponseFileSearchToolCall,
            ResponseComputerToolCall,
            ComputerCallOutput,
            ResponseFunctionWebSearch,
            ResponseFunctionToolCall,
            FunctionCallOutput,
            ResponseReasoningItem,
            ItemReference,
        ]
    ]
    tools: Optional[list[Union[FileSearchTool, FunctionTool, ComputerTool, WebSearchTool]]]
    custom_inputs: Optional[dict[str, Any]] = None


class ResponsesResponse(Response):
    custom_outputs: Optional[dict[str, Any]] = None


class ResponsesStreamEvent(BaseModel):
    TODO: str


RESPONSES_AGENT_INPUT_SCHEMA = _infer_schema_from_type_hint(ResponsesRequest)
RESPONSES_AGENT_OUTPUT_SCHEMA = _infer_schema_from_type_hint(ResponsesResponse)
RESPONSES_AGENT_INPUT_EXAMPLE = {"input": [{"role": "user", "content": "Hello!"}]}
