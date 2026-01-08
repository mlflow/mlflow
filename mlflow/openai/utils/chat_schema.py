import logging
from collections.abc import Iterable
from typing import Any

from mlflow.entities.span import LiveSpan
from mlflow.exceptions import MlflowException
from mlflow.tracing import set_span_chat_tools
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.types.chat import ChatTool, FunctionToolDefinition

_logger = logging.getLogger(__name__)


_RESPONSE_API_BUILT_IN_TOOLS = {
    "file_search",
    "computer_use_preview",
    "web_search_preview",
    "local_shell",
    "mcp",
    "code_interpreter",
    "image_generation",
}


def set_span_chat_attributes(span: LiveSpan, inputs: dict[str, Any], output: Any):
    # NB: This function is also used for setting chat attributes for ResponsesAgent tracing spans
    # (TODO: Add doc link). Therefore, the core logic should still run without openai package.
    try:
        if tools := _parse_tools(inputs):
            set_span_chat_tools(span, tools)
    except MlflowException:
        _logger.debug("Failed to set chat tools on span", exc_info=True)

    # Extract and set usage information if available
    if usage := _parse_usage(output):
        span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)


def _extract_tool_call_ids(output: Any) -> list[str]:
    tool_call_ids = []
    try:
        from openai.types.chat import ChatCompletion

        if isinstance(output, ChatCompletion):
            message = output.choices[0].message
            if tool_calls := getattr(message, "tool_calls", None):
                tool_call_ids.extend(tool_call.id for tool_call in tool_calls)
    except ImportError:
        pass

    if _is_responses_output(output):
        tool_call_ids.extend(
            call_id
            for output_item in output.output
            if (call_id := getattr(output_item, "call_id", None))
        )

    return tool_call_ids


def _is_responses_output(output: Any) -> bool:
    """
    Check whether the output is OpenAI Responses API output, or
    a response from the MLflow ResponsesAgent instance.
    """
    try:
        from openai.types.responses import Response

        if isinstance(output, Response):
            return True
    except ImportError:
        pass

    try:
        from mlflow.types.responses import ResponsesAgentResponse

        if ResponsesAgentResponse.model_validate(output):
            return True
    except Exception:
        pass

    return False


def _parse_tools(inputs: dict[str, Any]) -> list[ChatTool]:
    tools = inputs.get("tools", [])

    if tools is None or not isinstance(tools, Iterable):
        return []

    parsed_tools = []
    for tool in tools:
        tool_type = tool.get("type", "function")
        if tool_type == "function":
            if "function" in tool:
                # ChatCompletion API style
                parsed_tools.append(ChatTool(**tool))
            else:
                # Responses API style
                definition = {k: v for k, v in tool.items() if k != "type"}
                parsed_tools.append(
                    ChatTool(
                        type="function",
                        function=FunctionToolDefinition(**definition),
                    )
                )
        elif tool_type in _RESPONSE_API_BUILT_IN_TOOLS:
            parsed_tools.append(
                ChatTool(
                    type="function",
                    function=FunctionToolDefinition(
                        name=tool_type,
                    ),
                )
            )
        else:
            raise MlflowException(f"Unknown tool type: {tool_type}")

    return parsed_tools


def _parse_usage(output: Any) -> dict[str, Any] | None:
    """
    Parse token usage information from OpenAI response objects.

    Args:
        output: The response object from OpenAI API calls

    Returns:
        A dictionary containing token usage information.
    """
    if output is None:
        return None

    # Handle OpenAI ChatCompletion API response
    try:
        from openai.types.chat import ChatCompletion

        if isinstance(output, ChatCompletion) and (usage := output.usage):
            return {
                TokenUsageKey.INPUT_TOKENS: usage.prompt_tokens,
                TokenUsageKey.OUTPUT_TOKENS: usage.completion_tokens,
                TokenUsageKey.TOTAL_TOKENS: usage.total_tokens,
            }
    except ImportError:
        pass

    # Handle OpenAI Responses API response
    try:
        from openai.types.responses import Response

        if isinstance(output, Response) and (usage := output.usage):
            return {
                TokenUsageKey.INPUT_TOKENS: usage.input_tokens,
                TokenUsageKey.OUTPUT_TOKENS: usage.output_tokens,
                TokenUsageKey.TOTAL_TOKENS: usage.total_tokens,
            }
    except ImportError:
        pass

    return None
