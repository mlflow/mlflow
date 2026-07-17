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
    from mlflow.genai.judges.tools.registry import _judge_tool_registry

    tool_response_messages = []
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
        else:
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
    return tool_response_messages


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


def _remove_oldest_tool_call_pair(
    messages: list[Any],
) -> list[Any] | None:
    """Remove the oldest assistant message with tool calls and its corresponding tool responses.

    Works with any message type that has `role`, `tool_calls`, and `tool_call_id` attributes
    (e.g. ChatMessage, litellm.Message).
    """
    result = next(
        ((i, msg) for i, msg in enumerate(messages) if msg.role == "assistant" and msg.tool_calls),
        None,
    )
    if result is None:
        return None

    assistant_idx, assistant_msg = result
    modified = messages[:]
    modified.pop(assistant_idx)

    tool_call_ids = {tc.id if hasattr(tc, "id") else tc["id"] for tc in assistant_msg.tool_calls}
    return [
        msg for msg in modified if not (msg.role == "tool" and msg.tool_call_id in tool_call_ids)
    ]
