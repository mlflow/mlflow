from typing import Any, Literal, Optional

from openai.types.responses.response import Response

from mlflow.types.chat import BaseModel


class BaseRequestPayload(BaseModel):
    pass


class ResponseInputParam(BaseModel):
    pass


class FunctionTool(BaseModel):
    name: str
    parameters: dict[str, object]
    strict: bool
    type: Literal["function"]
    description: Optional[str] = None


class ResponsesRequest(BaseRequestPayload):
    input: list[ResponseInputParam]
    tools: Optional[list[FunctionTool]] = None
    custom_inputs: dict[str, Any]


class ResponsesResponse(Response):
    custom_outputs: dict[str, Any]


class ResponsesStreamEvent(BaseModel):
    pass
