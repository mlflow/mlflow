import json
from functools import lru_cache
from typing import Any

from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.tracing.constant import (
    TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH_DBX,
    TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH_OSS,
)
from mlflow.tracking._tracking_service.utils import get_tracking_uri
from mlflow.utils.uri import is_databricks_uri


def set_request_response_preview(trace_info: TraceInfo, trace_data: TraceData) -> None:
    """
    Set the request and response previews for the trace info.
    """
    # If request/response preview is already set by users via `mlflow.update_current_trace`,
    # we don't override it with the truncated version.
    if trace_info.request_preview is None:
        trace_info.request_preview = _get_truncated_preview(trace_data.request, role="user")
    if trace_info.response_preview is None:
        trace_info.response_preview = _get_truncated_preview(trace_data.response, role="assistant")


def _get_truncated_preview(request_or_response: str | None, role: str) -> str:
    """
    Truncate the request preview to fit the max length.
    """
    if request_or_response is None:
        return ""

    max_length = _get_max_length()

    content = None
    obj = None

    try:
        obj = json.loads(request_or_response)
    except json.JSONDecodeError:
        pass

    if obj is not None:
        if messages := _try_extract_messages(obj):
            msg = _get_last_message(messages, role=role)
            content = _get_text_content_from_message(msg)

    content = content or request_or_response

    if len(content) <= max_length:
        return content

    return content[: max_length - 3] + "..."


@lru_cache(maxsize=1)
def _get_max_length() -> int:
    tracking_uri = get_tracking_uri()
    return (
        TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH_DBX
        if is_databricks_uri(tracking_uri)
        else TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH_OSS
    )


def _try_extract_messages(obj: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not isinstance(obj, dict):
        return None

    # Check if the object contains messages with OpenAI ChatCompletion format
    if (messages := obj.get("messages")) and isinstance(messages, list):
        return [item for item in messages if _is_message(item)]

    # Check if the object contains a message in OpenAI ChatCompletion response format (choices)
    if (
        (choices := obj.get("choices"))
        and isinstance(choices, list)
        and len(choices) > 0
        and isinstance(choices[0], dict)
        and (msg := choices[0].get("message"))
        and _is_message(msg)
    ):
        return [msg]

    # Check if the object contains a message in OpenAI Responses API request format
    if (input := obj.get("input")) and isinstance(input, list):
        return [item for item in input if _is_message(item)]

    # Check if the object contains a message in OpenAI Responses API response format
    if (output := obj.get("output")) and isinstance(output, list):
        return [item for item in output if _is_message(item)]

    # Handle ResponsesAgent input, which contains OpenAI Responses request in 'request' key
    if "request" in obj:
        return _try_extract_messages(obj["request"])

    return None


def _is_message(item: Any) -> bool:
    return isinstance(item, dict) and "role" in item and "content" in item


def _get_last_message(messages: list[dict[str, Any]], role: str) -> dict[str, Any]:
    """
    Return last message with the given role.
    If the messages don't include a message with the given role, return the last one.
    """
    for message in reversed(messages):
        if message.get("role") == role:
            return message
    return messages[-1]


def _get_text_content_from_message(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        # content is a list of content parts
        for part in content:
            if isinstance(part, str):
                return part
            elif isinstance(part, dict) and part.get("type") in ["text", "output_text"]:
                return part.get("text")
    elif isinstance(content, str):
        return content
    return ""
