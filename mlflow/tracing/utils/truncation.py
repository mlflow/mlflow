import json
from typing import Any, Optional

from mlflow.tracing.constant import TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH


def truncate_request_response_preview(request_or_response: Optional[str], role: str) -> str:
    """
    Truncate the request preview to fit the max length.
    """
    if request_or_response is None:
        return ""

    if len(request_or_response) <= TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH:
        return request_or_response

    content = None
    if messages := _try_extract_messages(request_or_response):
        msg = _get_last_message(messages, role=role)
        content = _get_text_content_from_message(msg)

    content = content or request_or_response

    if len(content) <= TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH:
        return content

    return content[: TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH - 3] + "..."


def _try_extract_messages(obj_str: str) -> Optional[list[dict[str, Any]]]:
    try:
        obj = json.loads(obj_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None

    # Check if the object contains messages with OpenAI ChatCompletion format
    if messages := obj.get("messages"):
        return messages

    # Check if the object contains a message in OpenAI ChatCompletion response format (choices)
    if (choices := obj.get("choices")) and len(choices) > 0:
        return [choices[0].get("message")]

    return None


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
            elif isinstance(part, dict) and part.get("type") == "text":
                return part.get("text")
    elif isinstance(content, str):
        return content
    return ""
