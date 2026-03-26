"""Tool calling support for judge models."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from mlflow.entities.trace import Trace
    from mlflow.types.llm import ToolCall

from mlflow.environment_variables import MLFLOW_JUDGE_MAX_ITERATIONS
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.types import JudgeMessage
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
    tool_calls: list[dict[str, Any]],
    trace: Trace | None,
) -> list[JudgeMessage]:
    """
    Process tool calls and return tool response messages.

    Args:
        tool_calls: List of tool call dicts in OpenAI format
            (each has "id", "function": {"name": ..., "arguments": ...}).
        trace: Optional trace object for context.

    Returns:
        List of JudgeMessage objects containing tool responses.
    """
    from mlflow.genai.judges.tools.registry import _judge_tool_registry

    tool_response_messages = []
    for tool_call in tool_calls:
        tool_call_id = tool_call["id"]
        tool_call_function = tool_call["function"]
        tool_call_name = tool_call_function["name"]
        try:
            mlflow_tool_call = _create_tool_call(tool_call)
            result = _judge_tool_registry.invoke(tool_call=mlflow_tool_call, trace=trace)
        except Exception as e:
            tool_response_messages.append(
                _create_tool_response_message(
                    tool_call_id=tool_call_id,
                    tool_name=tool_call_name,
                    content=f"Error: {e!s}",
                )
            )
        else:
            if is_dataclass(result):
                result = asdict(result)
            result_json = json.dumps(result, default=str) if not isinstance(result, str) else result
            tool_response_messages.append(
                _create_tool_response_message(
                    tool_call_id=tool_call_id,
                    tool_name=tool_call_name,
                    content=result_json,
                )
            )
    return tool_response_messages


def _create_tool_call(tool_call_dict: dict[str, Any]) -> "ToolCall":
    """
    Create an MLflow ToolCall from an OpenAI-format tool call dict.

    Args:
        tool_call_dict: A dict with "id" and "function": {"name": ..., "arguments": ...}.

    Returns:
        An MLflow ToolCall object.
    """
    from mlflow.types.llm import ToolCall

    return ToolCall(
        id=tool_call_dict["id"],
        function={
            "name": tool_call_dict["function"]["name"],
            "arguments": tool_call_dict["function"]["arguments"],
        },
    )


def _create_tool_response_message(tool_call_id: str, tool_name: str, content: str) -> JudgeMessage:
    """
    Create a tool response message.

    Args:
        tool_call_id: The ID of the tool call being responded to.
        tool_name: The name of the tool that was invoked.
        content: The content to include in the response.

    Returns:
        A JudgeMessage representing the tool response.
    """
    return JudgeMessage(
        tool_call_id=tool_call_id,
        role="tool",
        name=tool_name,
        content=content,
    )
