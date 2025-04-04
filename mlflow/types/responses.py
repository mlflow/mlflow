import json
import warnings
from typing import Any, Optional, Union

from pydantic import ConfigDict

from mlflow.types.chat import BaseModel
from mlflow.types.responses_helpers import (
    BaseRequestPayload,
    FunctionCallOutput,
    Message,
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseErrorEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseTextAnnotationDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    Tools,
)
from mlflow.types.schema import Schema
from mlflow.types.type_hints import _infer_schema_from_type_hint
from mlflow.utils.pydantic_utils import model_validator


class ResponsesRequest(BaseRequestPayload, Tools):
    input: list[
        Union[
            Message,
            ResponseOutputMessage,
            ResponseFunctionToolCall,
            FunctionCallOutput,
            dict[str, Any],
        ]
    ]
    custom_inputs: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def check_input(self) -> "ResponsesRequest":
        for input in self.input:
            if isinstance(
                input,
                (
                    Message,
                    ResponseOutputMessage,
                    ResponseFunctionToolCall,
                    FunctionCallOutput,
                ),
            ):
                continue
            if isinstance(input, dict):
                if "type" not in input:
                    raise ValueError("dict must have a type key")
                if input["type"] not in {
                    "file_search_call",
                    "computer_call",
                    "web_search_call",
                    "item_reference",
                }:
                    # if they are one of these types, we will not validate them
                    # will allow the model to fail w/ bad_request w/ these message types
                    raise ValueError(f"Invalid type: {input['type']}")
        return self


class ResponsesResponse(Response):
    custom_outputs: Optional[dict[str, Any]] = None


class ResponsesStreamEvent(BaseModel):
    # pydantic model that allows for all other streaming event types to pass type validation
    model_config = ConfigDict(extra="allow")
    type: str
    custom_outputs: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def check_type(self) -> "ResponsesStreamEvent":
        if self.type == "response.output_item.added":
            ResponseOutputItemAddedEvent(**self.model_dump_compat())
        elif self.type == "response.output_item.done":
            ResponseOutputItemDoneEvent(**self.model_dump_compat())
        elif self.type == "response.content_part.added":
            ResponseContentPartAddedEvent(**self.model_dump_compat())
        elif self.type == "response.content_part.done":
            ResponseContentPartDoneEvent(**self.model_dump_compat())
        elif self.type == "response.output_text.delta":
            ResponseTextDeltaEvent(**self.model_dump_compat())
        elif self.type == "response.output_text.done":
            ResponseTextDoneEvent(**self.model_dump_compat())
        elif self.type == "response.output_text.annotation.added":
            ResponseTextAnnotationDeltaEvent(**self.model_dump_compat())
        elif self.type == "response.function_call_arguments.delta":
            ResponseFunctionCallArgumentsDeltaEvent(**self.model_dump_compat())
        elif self.type == "response.function_call_arguments.done":
            ResponseFunctionCallArgumentsDoneEvent(**self.model_dump_compat())
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
            "response.refusal.delta",
            "response.refusal.done",
            "response.file_search_call.in_progress",
            "response.file_search_call.searching",
            "response.file_search_call.completed",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
        }:
            warnings.warn(f"Invalid type: {self.type}.")
        return self


properties = _infer_schema_from_type_hint(ResponsesRequest).to_dict()[0]["properties"]
formatted_properties = [{**prop, "name": name} for name, prop in properties.items()]
RESPONSES_AGENT_INPUT_SCHEMA = Schema.from_json(json.dumps(formatted_properties))
RESPONSES_AGENT_OUTPUT_SCHEMA = _infer_schema_from_type_hint(ResponsesResponse)
RESPONSES_AGENT_INPUT_EXAMPLE = {"input": [{"role": "user", "content": "Hello!"}]}
