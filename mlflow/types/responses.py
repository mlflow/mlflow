from typing import Any, Literal, Optional

from openai.types.responses.response import Response

from mlflow.types.chat import BaseModel
from mlflow.types.type_hints import _infer_schema_from_type_hint


class BaseRequestPayload(BaseModel):
    TODO: str


class ResponseInputParam(BaseModel):
    TODO: str


class FunctionTool(BaseModel):
    name: str
    parameters: dict[str, Any]
    strict: bool
    type: Literal["function"] = "function"
    description: Optional[str] = None


class ResponsesRequest(BaseRequestPayload):
    input: list[ResponseInputParam]
    tools: Optional[list[FunctionTool]] = None
    # custom_inputs: Optional[dict[str, Any]] = None


class ResponsesResponse(Response):
    custom_outputs: Optional[dict[str, Any]] = None


class ResponsesStreamEvent(BaseModel):
    TODO: str


RESPONSES_AGENT_INPUT_SCHEMA = _infer_schema_from_type_hint(ResponsesRequest)
RESPONSES_AGENT_OUTPUT_SCHEMA = _infer_schema_from_type_hint(ResponsesResponse)
RESPONSES_AGENT_INPUT_EXAMPLE = {"input": [{"role": "user", "content": "Hello!"}]}
