import json
from typing import Any, Optional

from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.tracing.constant import TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH


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


def _get_truncated_preview(request_or_response: Optional[str], role: str) -> str:
    """
    Truncate the request preview to fit the max length.
    """
    if request_or_response is None:
        return ""

    if len(request_or_response) <= TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH:
        return request_or_response

    content = None

    # Parse JSON serialized request/response
    try:
        obj = json.loads(request_or_response)
    except json.JSONDecodeError:
        obj = None

    if messages := _try_extract_messages(obj):
        msg = _get_last_message(messages, role=role)
        content = _get_text_content_from_message(msg)

    content = content or request_or_response

    if len(content) <= TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH:
        return content

    return content[: TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH - 3] + "..."


def _try_extract_messages(obj: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    if not isinstance(obj, dict):
        return None

    # Check if the object contains messages with OpenAI ChatCompletion format
    if messages := obj.get("messages"):
        return [item for item in messages if _is_message(item)]

    # Check if the object contains a message in OpenAI ChatCompletion response format (choices)
    if (choices := obj.get("choices")) and len(choices) > 0:
        return [choices[0].get("message")]

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
    if not isinstance(item, dict):
        return False
    return "role" in item and "content" in item


def _get_last_message(messages: list[dict[str, Any]], role: str) -> Optional[dict[str, Any]]:
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
