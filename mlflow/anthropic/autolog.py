import functools
import logging
from typing import Any

import mlflow.anthropic
from mlflow.anthropic.chat import convert_tool_to_mlflow_chat_tool
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.utils import (
    construct_full_inputs,
    set_span_chat_tools,
    set_span_model_attribute,
)
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def patched_claude_sdk_init(original, self, options=None):
    """Wrap query/receive_messages to capture messages and inject a Stop hook
    that builds an MLflow trace from the accumulated conversation.

    Args:
        original: The original ``ClaudeSDKClient.__init__`` method.
        self: The ``ClaudeSDKClient`` instance being initialized.
        options: Optional ``ClaudeAgentOptions`` forwarded to the original init.

    Returns:
        The result of the original ``__init__`` call.
    """
    try:
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
        from claude_agent_sdk.types import ResultMessage, UserMessage

        result = original(self, options)

        messages_buffer: list[Any] = []

        # Wrap query() to capture user prompts — query() sends the prompt to
        # the CLI subprocess but doesn't echo it through receive_messages
        original_query = self.query

        @functools.wraps(original_query)
        async def wrapped_query(prompt, *args, **kwargs):
            if isinstance(prompt, str):
                messages_buffer.append(UserMessage(content=prompt))
            return await original_query(prompt, *args, **kwargs)

        self.query = wrapped_query

        # Wrap receive_messages() to accumulate all streamed SDK messages
        original_receive_messages = self.receive_messages

        @functools.wraps(original_receive_messages)
        async def wrapped_receive_messages(*args, **kwargs):
            async for message in original_receive_messages(*args, **kwargs):
                messages_buffer.append(message)
                yield message

        self.receive_messages = wrapped_receive_messages

        # Wrap receive_response() to capture ResultMessage, which contains
        # token usage and duration but is only yielded by receive_response()
        # (not by receive_messages())
        original_receive_response = self.receive_response

        @functools.wraps(original_receive_response)
        async def wrapped_receive_response(*args, **kwargs):
            async for message in original_receive_response(*args, **kwargs):
                if isinstance(message, ResultMessage):
                    messages_buffer.append(message)
                yield message

        self.receive_response = wrapped_receive_response

        # Inject a Stop hook that builds the trace from accumulated messages
        async def stop_hook(input_data, tool_use_id, context):
            from mlflow.claude_code.hooks import get_hook_response
            from mlflow.utils.autologging_utils import autologging_is_disabled

            if autologging_is_disabled("anthropic"):
                return get_hook_response()

            try:
                from mlflow.claude_code.tracing import process_sdk_messages

                session_id = input_data.get("session_id")
                trace = process_sdk_messages(list(messages_buffer), session_id)

                if trace is not None:
                    return get_hook_response()
                return get_hook_response(
                    error="Failed to process SDK messages, check "
                    ".claude/mlflow/claude_tracing.log for details",
                )
            except Exception as e:
                _logger.debug("Error in SDK stop hook: %s", e, exc_info=True)
                return get_hook_response(error=str(e))
            finally:
                messages_buffer.clear()

        if self.options is None:
            self.options = ClaudeAgentOptions()
        if self.options.hooks is None:
            self.options.hooks = {}
        self.options.hooks.setdefault("Stop", []).append(HookMatcher(hooks=[stop_hook]))

        return result

    except Exception as e:
        _logger.debug("Error in patched_claude_sdk_init: %s", e, exc_info=True)
        return original(self, options)


def patched_class_call(original, self, *args, **kwargs):
    with TracingSession(original, self, args, kwargs) as manager:
        output = original(self, *args, **kwargs)
        manager.output = output
        return output


async def async_patched_class_call(original, self, *args, **kwargs):
    async with TracingSession(original, self, args, kwargs) as manager:
        output = await original(self, *args, **kwargs)
        manager.output = output
        return output


class TracingSession:
    """Context manager for handling MLflow spans in both sync and async contexts."""

    def __init__(self, original, instance, args, kwargs):
        self.original = original
        self.instance = instance
        self.inputs = construct_full_inputs(original, instance, *args, **kwargs)

        # These attributes are set outside the constructor.
        self.span = None
        self.output = None

    def __enter__(self):
        return self._enter_impl()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        return self._enter_impl()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)

    def _enter_impl(self):
        config = AutoLoggingConfig.init(flavor_name=mlflow.anthropic.FLAVOR_NAME)

        if config.log_traces:
            self.span = start_span_no_context(
                name=f"{self.instance.__class__.__name__}.{self.original.__name__}",
                span_type=_get_span_type(self.original.__name__),
                inputs=self.inputs,
                attributes={SpanAttributeKey.MESSAGE_FORMAT: "anthropic"},
            )
            _set_tool_attribute(self.span, self.inputs)

        return self

    def _exit_impl(self, exc_type, exc_val, exc_tb) -> None:
        if self.span:
            if exc_val:
                self.span.record_exception(exc_val)

            set_span_model_attribute(self.span, self.inputs)
            _set_token_usage_attribute(self.span, self.output)
            self.span.end(outputs=self.output)


def _get_span_type(task_name: str) -> str:
    # Anthropic has a few APIs in beta, e.g., count_tokens.
    # Once they are stable, we can add them to the mapping.
    span_type_mapping = {
        "create": SpanType.CHAT_MODEL,
    }
    return span_type_mapping.get(task_name, SpanType.UNKNOWN)


def _set_tool_attribute(span: LiveSpan, inputs: dict[str, Any]):
    if (tools := inputs.get("tools")) is not None:
        try:
            tools = [convert_tool_to_mlflow_chat_tool(tool) for tool in tools]
            set_span_chat_tools(span, tools)
        except Exception as e:
            _logger.debug(f"Failed to set tools for {span}. Error: {e}")


def _set_token_usage_attribute(span: LiveSpan, output: Any):
    try:
        if usage := _parse_usage(output):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)
    except Exception as e:
        _logger.debug(f"Failed to set token usage for {span}. Error: {e}")


def _parse_usage(output: Any) -> dict[str, int] | None:
    try:
        if usage := getattr(output, "usage", None):
            return {
                TokenUsageKey.INPUT_TOKENS: usage.input_tokens,
                TokenUsageKey.OUTPUT_TOKENS: usage.output_tokens,
                TokenUsageKey.TOTAL_TOKENS: usage.input_tokens + usage.output_tokens,
            }
    except Exception as e:
        _logger.debug(f"Failed to parse token usage from output: {e}")
    return None
