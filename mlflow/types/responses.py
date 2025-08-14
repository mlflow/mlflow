from mlflow.utils.pydantic_utils import IS_PYDANTIC_V2_OR_NEWER

if not IS_PYDANTIC_V2_OR_NEWER:
    raise ImportError(
        "mlflow.types.responses is not supported in Pydantic v1. "
        "Please upgrade to Pydantic v2 or newer."
    )
import json
from typing import Any

from pydantic import ConfigDict, model_validator

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
)

__all__ = [
    "ResponsesAgentRequest",
    "ResponsesAgentResponse",
    "ResponsesAgentStreamEvent",
]

from mlflow.types.schema import Schema
from mlflow.types.type_hints import _infer_schema_from_type_hint
from mlflow.utils.autologging_utils.logging_and_warnings import (
    MlflowEventsAndWarningsBehaviorGlobally,
)


class ResponsesAgentRequest(BaseRequestPayload):
    """Request object for ResponsesAgent.

    Args:
        input: List of simple `role` and `content` messages or output items. See examples at
            https://mlflow.org/docs/latest/llms/responses-agent-intro/#testing-out-your-agent and
            https://mlflow.org/docs/latest/llms/responses-agent-intro/#creating-agent-output.
        custom_inputs (Dict[str, Any]): An optional param to provide arbitrary additional context
            to the model. The dictionary values must be JSON-serializable.
            **Optional** defaults to ``None``
        context (:py:class:`mlflow.types.agent.ChatContext`): The context to be used in the chat
            endpoint. Includes conversation_id and user_id. **Optional** defaults to ``None``
    """

    input: list[Message | OutputItem]
    custom_inputs: dict[str, Any] | None = None
    context: ChatContext | None = None


class ResponsesAgentResponse(Response):
    """Response object for ResponsesAgent.

    Args:
        output: List of output items. See examples at
            https://mlflow.org/docs/latest/llms/responses-agent-intro/#creating-agent-output.
        reasoning: Reasoning parameters
        usage: Usage information
        custom_outputs (Dict[str, Any]): An optional param to provide arbitrary additional context
            from the model. The dictionary values must be JSON-serializable. **Optional**, defaults
            to ``None``
    """

    custom_outputs: dict[str, Any] | None = None


class ResponsesAgentStreamEvent(BaseModel):
    """Stream event for ResponsesAgent.
    See examples at https://mlflow.org/docs/latest/llms/responses-agent-intro/#streaming-agent-output

    Args:
        type (str): Type of the stream event
        custom_outputs (Dict[str, Any]): An optional param to provide arbitrary additional context
            from the model. The dictionary values must be JSON-serializable. **Optional**, defaults
            to ``None``
    """

    model_config = ConfigDict(extra="allow")
    type: str
    custom_outputs: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check_type(self) -> "ResponsesAgentStreamEvent":
        type = self.type
        if type == "response.output_item.done":
            ResponseOutputItemDoneEvent(**self.model_dump_compat())
        elif type == "response.output_text.delta":
            ResponseTextDeltaEvent(**self.model_dump_compat())
        elif type == "response.output_text.annotation.added":
            ResponseTextAnnotationDeltaEvent(**self.model_dump_compat())
        elif type == "error":
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
    properties = _infer_schema_from_type_hint(ResponsesAgentRequest).to_dict()[0]["properties"]
    formatted_properties = [{**prop, "name": name} for name, prop in properties.items()]
    RESPONSES_AGENT_INPUT_SCHEMA = Schema.from_json(json.dumps(formatted_properties))
    RESPONSES_AGENT_OUTPUT_SCHEMA = _infer_schema_from_type_hint(ResponsesAgentResponse)
RESPONSES_AGENT_INPUT_EXAMPLE = {"input": [{"role": "user", "content": "Hello!"}]}
