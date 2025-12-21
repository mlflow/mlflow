"""Tool calling support for judge models."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import litellm

    from mlflow.entities.trace import Trace
    from mlflow.types.llm import ToolCall


def _process_tool_calls(
    tool_calls: list["litellm.ChatCompletionMessageToolCall"],
    trace: Trace | None,
) -> list["litellm.Message"]:
    """
    Process tool calls and return tool response messages.

    Args:
        tool_calls: List of tool calls from the LLM response.
        trace: Optional trace object for context.

    Returns:
        List of litellm Message objects containing tool responses.
    """
    from mlflow.genai.judges.tools.registry import _judge_tool_registry

    tool_response_messages = []
    for tool_call in tool_calls:
        try:
            mlflow_tool_call = _create_mlflow_tool_call_from_litellm(litellm_tool_call=tool_call)
            result = _judge_tool_registry.invoke(tool_call=mlflow_tool_call, trace=trace)
        except Exception as e:
            tool_response_messages.append(
                _create_litellm_tool_response_message(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    content=f"Error: {e!s}",
                )
            )
        else:
            if is_dataclass(result):
                result = asdict(result)
            result_json = json.dumps(result, default=str) if not isinstance(result, str) else result
            tool_response_messages.append(
                _create_litellm_tool_response_message(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    content=result_json,
                )
            )
    return tool_response_messages


def _create_mlflow_tool_call_from_litellm(
    litellm_tool_call: "litellm.ChatCompletionMessageToolCall",
) -> "ToolCall":
    """
    Create an MLflow ToolCall from a LiteLLM tool call.

    Args:
        litellm_tool_call: The LiteLLM ChatCompletionMessageToolCall object.

    Returns:
        An MLflow ToolCall object.
    """
    from mlflow.types.llm import ToolCall

    return ToolCall(
        id=litellm_tool_call.id,
        function={
            "name": litellm_tool_call.function.name,
            "arguments": litellm_tool_call.function.arguments,
        },
    )


def _create_litellm_tool_response_message(
    tool_call_id: str, tool_name: str, content: str
) -> "litellm.Message":
    """
    Create a tool response message for LiteLLM.

    Args:
        tool_call_id: The ID of the tool call being responded to.
        tool_name: The name of the tool that was invoked.
        content: The content to include in the response.

    Returns:
        A litellm.Message object representing the tool response message.
    """
    import litellm

    return litellm.Message(
        tool_call_id=tool_call_id,
        role="tool",
        name=tool_name,
        content=content,
    )
