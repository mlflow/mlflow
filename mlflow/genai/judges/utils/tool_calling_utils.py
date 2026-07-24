"""Tool calling support for judge models."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from mlflow.entities.trace import Trace
    from mlflow.types.llm import ChatMessage, ToolCall

from mlflow.environment_variables import MLFLOW_JUDGE_MAX_ITERATIONS
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import REQUEST_LIMIT_EXCEEDED

# Attribute used to tag an injected multimodal user-turn with the tool_call_id that
# produced it, so context-window pruning can drop it together with its tool-call pair.
# The user-turn has no tool_call_id of its own, so without this tag pruning would
# orphan it and break strict role alternation on overflow.
IMAGE_TURN_TOOL_CALL_ID_ATTR = "_mlflow_image_turn_tool_call_id"


def _raise_iteration_limit_exceeded(max_iterations: int) -> NoReturn:
    """Raise an exception when the agentic loop iteration limit is exceeded.

    Args:
        max_iterations: The maximum number of iterations that was exceeded.

    Raises:
        MlflowException: Always raises with REQUEST_LIMIT_EXCEEDED error code.
    """
    raise MlflowException(
        f"Completion iteration limit of {max_iterations} exceeded. "
        f"This usually indicates the model is not powerful enough to effectively "
        f"analyze the trace. Consider using a more intelligent/powerful model. "
        f"In rare cases, for very complex traces where a large number of completion "
        f"iterations might be required, you can increase the number of iterations by "
        f"modifying the {MLFLOW_JUDGE_MAX_ITERATIONS.name} environment variable.",
        error_code=REQUEST_LIMIT_EXCEEDED,
    )


def _process_tool_calls(
    tool_calls: list[ToolCall],
    trace: Trace | None,
) -> list[ChatMessage]:
    """
    Process tool calls and return tool response messages.

    Args:
        tool_calls: List of ToolCall objects from the model response.
        trace: Optional trace object for context.

    Returns:
        List of ChatMessage objects containing tool responses.
    """
    from mlflow.genai.judges.tools.get_span_image import SpanImageResult
    from mlflow.genai.judges.tools.registry import _judge_tool_registry

    # OpenAI requires every role="tool" response for an assistant turn's tool_calls to
    # be consecutive, immediately after the assistant message. So collect all tool
    # responses first and append the injected image user-turn(s) only after the full
    # tool-response block, even when one assistant turn batches multiple tool calls.
    tool_response_messages = []
    image_turn_messages = []
    for tool_call in tool_calls:
        try:
            result = _judge_tool_registry.invoke(tool_call=tool_call, trace=trace)
        except Exception as e:
            tool_response_messages.append(
                _create_tool_response_message(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    content=f"Error: {e!s}",
                )
            )
            continue

        # Image blocks are rejected inside role="tool" messages, so emit a text ack
        # to satisfy the tool_call_id and defer the image to a user turn (see above).
        if isinstance(result, SpanImageResult):
            tool_response_messages.append(
                _create_tool_response_message(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    content=(
                        f"Image for span {result.span_id} fetched; it is shown in the "
                        "following user message. Inspect it to answer."
                    ),
                )
            )
            image_turn_messages.append(
                _create_image_turn_message(tool_call_id=tool_call.id, result=result)
            )
            continue

        if is_dataclass(result):
            result = asdict(result)
        result_json = (
            json.dumps(result, default=str, ensure_ascii=False)
            if not isinstance(result, str)
            else result
        )
        tool_response_messages.append(
            _create_tool_response_message(
                tool_call_id=tool_call.id,
                tool_name=tool_call.function.name,
                content=result_json,
            )
        )
    return tool_response_messages + image_turn_messages


def _create_tool_response_message(tool_call_id: str, tool_name: str, content: str) -> "ChatMessage":
    """
    Create a tool response message.

    Args:
        tool_call_id: The ID of the tool call being responded to.
        tool_name: The name of the tool that was invoked.
        content: The content to include in the response.

    Returns:
        A ChatMessage representing the tool response.
    """
    from mlflow.types.llm import ChatMessage

    return ChatMessage(
        tool_call_id=tool_call_id,
        role="tool",
        name=tool_name,
        content=content,
    )


def _create_image_turn_message(tool_call_id: str, result: Any) -> "ChatMessage":
    """Build the user message that delivers a fetched image, tagged with its tool_call_id.

    The tag lets a user message (which has no tool_call_id of its own) still be paired
    with the tool call that produced it when trimming messages.
    """
    from mlflow.types.llm import ChatMessage

    message = ChatMessage(
        role="user",
        content=[
            {"type": "text", "text": f"Fetched image for span {result.span_id}:"},
            {"type": "image_url", "image_url": {"url": result.data_url}},
        ],
    )
    _tag_image_turn(message, tool_call_id)
    return message


class _ImageTurnDict(dict):
    """A multimodal user-message dict whose tool_call_id rides as an instance attribute.

    Storing the tool_call_id as an attribute rather than a dict key keeps it out of the
    message sent to the model provider (strict OpenAI-compatible endpoints 400 on unknown
    message keys) while still letting it be paired with its tool call when trimming.
    """


def _tag_image_turn(message: Any, tool_call_id: str) -> None:
    """Tag a message as an injected image user-turn belonging to ``tool_call_id``.

    The tag is always an instance attribute (never a dict key), so it stays out of the
    message sent to the model provider. Best-effort: if the message type rejects the
    extra attribute, the turn simply stays in place when trimming, which is safe.
    """
    try:
        object.__setattr__(message, IMAGE_TURN_TOOL_CALL_ID_ATTR, tool_call_id)
    except (AttributeError, TypeError):
        pass


def _to_litellm_image_turn(message: Any) -> _ImageTurnDict:
    """Convert an image-turn ChatMessage into an _ImageTurnDict.

    litellm.Message types content as ``str`` and rejects the multimodal list, so the
    image turn is sent as a dict instead.
    """
    turn = _ImageTurnDict(role=message.role, content=message.content)
    _tag_image_turn(turn, _get_image_turn_tool_call_id(message))
    return turn


def _get_image_turn_tool_call_id(message: Any) -> str | None:
    """Return the tool_call_id an injected image user-turn belongs to, or None if untagged."""
    return getattr(message, IMAGE_TURN_TOOL_CALL_ID_ATTR, None)


def _get_message_field(message: Any, field: str) -> Any:
    if isinstance(message, dict):
        return message.get(field)
    return getattr(message, field, None)


def _remove_oldest_tool_call_pair(
    messages: list[Any],
) -> list[Any] | None:
    """Remove the oldest assistant message with tool calls and its corresponding tool responses.

    Works with any message type that has `role`, `tool_calls`, and `tool_call_id` attributes
    (e.g. ChatMessage, litellm.Message) as well as plain dict messages (the litellm image
    turn). Any injected image user-turn (a role="user" message tagged with a removed
    tool_call_id) is dropped alongside its pair so it is not orphaned.
    """
    result = next(
        (
            (i, msg)
            for i, msg in enumerate(messages)
            if _get_message_field(msg, "role") == "assistant"
            and _get_message_field(msg, "tool_calls")
        ),
        None,
    )
    if result is None:
        return None

    assistant_idx, assistant_msg = result
    modified = messages[:]
    modified.pop(assistant_idx)

    tool_call_ids = {
        tc.id if hasattr(tc, "id") else tc["id"]
        for tc in _get_message_field(assistant_msg, "tool_calls")
    }
    return [
        msg
        for msg in modified
        if not (
            _get_message_field(msg, "role") == "tool"
            and _get_message_field(msg, "tool_call_id") in tool_call_ids
        )
        and _get_image_turn_tool_call_id(msg) not in tool_call_ids
    ]
