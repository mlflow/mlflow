import json
import warnings
from typing import Any, Optional, Union

from pydantic import ConfigDict

from mlflow.types.agent import ChatContext
from mlflow.types.chat import BaseModel
from mlflow.types.responses_helpers import (
    BaseRequestPayload,
    Message,
    OutputItem,
    Response,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextAnnotationDeltaEvent,
    ResponseTextDeltaEvent,
    Tools,
)
from mlflow.types.schema import Schema
from mlflow.types.type_hints import _infer_schema_from_type_hint
from mlflow.utils.pydantic_utils import model_validator


class ResponsesRequest(BaseRequestPayload, Tools):
    input: list[Union[Message, OutputItem]]
    custom_inputs: Optional[dict[str, Any]] = None
    context: Optional[ChatContext] = None


class ResponsesResponse(Response):
    custom_outputs: Optional[dict[str, Any]] = None


class ResponsesStreamEvent(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    custom_outputs: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def check_type(self) -> "ResponsesStreamEvent":
        if self.type == "response.output_item.done":
            ResponseOutputItemDoneEvent(**self.model_dump_compat())
        elif self.type == "response.output_text.delta":
            ResponseTextDeltaEvent(**self.model_dump_compat())
        elif self.type == "response.output_text.annotation.added":
            ResponseTextAnnotationDeltaEvent(**self.model_dump_compat())
        elif self.type == "response.error":
            ResponseErrorEvent(**self.model_dump_compat())
        elif self.type == "response.completed":
            ResponseCompletedEvent(**self.model_dump_compat())
        elif self.type not in {
            "response.created",
            "response.in_progress",
            "response.completed",
            "response.failed",
            "response.incomplete",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.done",
            "response.output_item.added",
            "response.refusal.delta",
            "response.refusal.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.file_search_call.in_progress",
            "response.file_search_call.searching",
            "response.file_search_call.completed",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.error",
        }:
            warnings.warn(f"Invalid type: {self.type} for ResponsesStreamEvent.")
        return self


properties = _infer_schema_from_type_hint(ResponsesRequest).to_dict()[0]["properties"]
formatted_properties = [{**prop, "name": name} for name, prop in properties.items()]
RESPONSES_AGENT_INPUT_SCHEMA = Schema.from_json(json.dumps(formatted_properties))
RESPONSES_AGENT_OUTPUT_SCHEMA = _infer_schema_from_type_hint(ResponsesResponse)
RESPONSES_AGENT_INPUT_EXAMPLE = {"input": [{"role": "user", "content": "Hello!"}]}
