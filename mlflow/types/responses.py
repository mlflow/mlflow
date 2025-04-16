from mlflow.utils.pydantic_utils import IS_PYDANTIC_V2_OR_NEWER, model_validator

if not IS_PYDANTIC_V2_OR_NEWER:
    raise ImportError(
        "mlflow.types.responses is not supported in Pydantic v1. "
        "Please upgrade to Pydantic v2 or newer."
    )

import json
from typing import Any, Optional, Union

from pydantic import ConfigDict

from mlflow.types.agent import ChatContext
from mlflow.types.chat import BaseModel
from mlflow.types.responses_helpers import (
    # Re-export all other classes
    Annotation,
    AnnotationFileCitation,
    AnnotationFilePath,
    AnnotationURLCitation,
    BaseRequestPayload,
    Content,
    FunctionCallOutput,
    FunctionTool,
    IncompleteDetails,
    InputTokensDetails,
    Message,
    OutputItem,
    OutputTokensDetails,
    ReasoningParams,
    Response,
    ResponseCompletedEvent,
    ResponseError,
    ResponseErrorEvent,
    ResponseFunctionToolCall,
    ResponseInputTextParam,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextAnnotationDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
    Status,
    Summary,
    Tool,
    ToolChoice,
    ToolChoiceFunction,
    Truncation,
)

__all__ = [
    # Classes defined in this file
    "ResponsesRequest",
    "ResponsesResponse",
    "ResponsesStreamEvent",
    # Re-exported classes from responses_helpers
    "Annotation",
    "AnnotationFileCitation",
    "AnnotationFilePath",
    "AnnotationURLCitation",
    "BaseRequestPayload",
    "Content",
    "FunctionCallOutput",
    "FunctionTool",
    "IncompleteDetails",
    "InputTokensDetails",
    "Message",
    "OutputItem",
    "OutputTokensDetails",
    "ReasoningParams",
    "Response",
    "ResponseCompletedEvent",
    "ResponseError",
    "ResponseErrorEvent",
    "ResponseFunctionToolCall",
    "ResponseInputTextParam",
    "ResponseOutputItemDoneEvent",
    "ResponseOutputMessage",
    "ResponseOutputRefusal",
    "ResponseOutputText",
    "ResponseReasoningItem",
    "ResponseTextAnnotationDeltaEvent",
    "ResponseTextDeltaEvent",
    "ResponseUsage",
    "Status",
    "Summary",
    "Tool",
    "ToolChoice",
    "ToolChoiceFunction",
    "Truncation",
]

from mlflow.types.schema import Schema
from mlflow.types.type_hints import _infer_schema_from_type_hint
from mlflow.utils.autologging_utils.logging_and_warnings import (
    MlflowEventsAndWarningsBehaviorGlobally,
)


class ResponsesRequest(BaseRequestPayload):
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
        type = self.type
        if type == "response.output_item.done":
            ResponseOutputItemDoneEvent(**self.model_dump_compat())
        elif type == "response.output_text.delta":
            ResponseTextDeltaEvent(**self.model_dump_compat())
        elif type == "response.output_text.annotation.added":
            ResponseTextAnnotationDeltaEvent(**self.model_dump_compat())
        elif type == "response.error":
            ResponseErrorEvent(**self.model_dump_compat())
        elif type == "response.completed":
            ResponseCompletedEvent(**self.model_dump_compat())
        """
        unvalidated types: {
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
        }
        """
        return self


with MlflowEventsAndWarningsBehaviorGlobally(
    reroute_warnings=False,
    disable_event_logs=True,
    disable_warnings=True,
):
    properties = _infer_schema_from_type_hint(ResponsesRequest).to_dict()[0]["properties"]
    formatted_properties = [{**prop, "name": name} for name, prop in properties.items()]
    RESPONSES_AGENT_INPUT_SCHEMA = Schema.from_json(json.dumps(formatted_properties))
    RESPONSES_AGENT_OUTPUT_SCHEMA = _infer_schema_from_type_hint(ResponsesResponse)
RESPONSES_AGENT_INPUT_EXAMPLE = {"input": [{"role": "user", "content": "Hello!"}]}
