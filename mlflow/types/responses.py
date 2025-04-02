import json
from typing import Any, Optional, Union

from pydantic import model_validator

from mlflow.types.responses_helpers import (
    BaseRequestPayload,
    FunctionCallOutput,
    Message,
    Response,
    ResponseCompletedEvent,
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
    StreamCatchAllEvent,
    Tools,
)
from mlflow.types.schema import Schema
from mlflow.types.type_hints import _infer_schema_from_type_hint


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


ResponsesStreamEvent = Union[
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseTextAnnotationDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseErrorEvent,
    ResponseCompletedEvent,
    StreamCatchAllEvent,
]

x = _infer_schema_from_type_hint(ResponsesRequest).to_dict()
properties = x[0]["properties"]
temp = []
for p in properties:
    properties[p]["name"] = p
    temp.append(properties[p])

RESPONSES_AGENT_INPUT_SCHEMA = Schema.from_json(json.dumps(temp))

RESPONSES_AGENT_OUTPUT_SCHEMA = _infer_schema_from_type_hint(ResponsesResponse)
RESPONSES_AGENT_INPUT_EXAMPLE = {"input": [{"role": "user", "content": "Hello!"}]}
