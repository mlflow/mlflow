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


def _build_trace(messages_buffer: list[Any], session_id: str | None = None) -> None:
    """Build an MLflow trace from accumulated SDK messages."""
    from mlflow.utils.autologging_utils import autologging_is_disabled

    if autologging_is_disabled("anthropic"):
        return

    try:
        from mlflow.claude_code.tracing import process_sdk_messages

        process_sdk_messages(list(messages_buffer), session_id)
    except Exception as e:
        _logger.debug("Error building trace from SDK messages: %s", e, exc_info=True)
    finally:
        messages_buffer.clear()


def patched_claude_sdk_init(original, self, options=None):
    """Wrap query/receive_messages/receive_response to capture messages and
    build an MLflow trace when the conversation completes.

    The SDK fires the Stop hook BEFORE yielding ResultMessage (which carries
    token usage and duration).  Therefore we build the trace when
    receive_response() is fully consumed — at that point the buffer contains
    all conversation messages plus ResultMessage.  A Stop hook is still
    injected as a fallback for code paths that never call receive_response().
    """
    try:
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
        from claude_agent_sdk.types import ResultMessage, UserMessage

        result = original(self, options)

        messages_buffer: list[Any] = []
        trace_built = False
        receiving_response = False

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

        # Wrap receive_response() to capture all messages including
        # ResultMessage (which carries token usage + duration).  Build
        # the trace once the generator is exhausted so that ResultMessage
        # is guaranteed to be in the buffer.
        original_receive_response = self.receive_response

        @functools.wraps(original_receive_response)
        async def wrapped_receive_response(*args, **kwargs):
            nonlocal trace_built, receiving_response
            receiving_response = True
            try:
                async for message in original_receive_response(*args, **kwargs):
                    if isinstance(message, ResultMessage):
                        messages_buffer.append(message)
                    yield message
            finally:
                receiving_response = False

            # Generator exhausted — all messages including ResultMessage are
            # now in the buffer.  Build the trace here.
            result_msg = next(
                (m for m in messages_buffer if isinstance(m, ResultMessage)), None
            )
            session_id = getattr(result_msg, "session_id", None) if result_msg else None
            _build_trace(messages_buffer, session_id)
            trace_built = True

        self.receive_response = wrapped_receive_response

        # Stop hook fallback — only used if receive_response() was never
        # consumed (e.g. user only called receive_messages()).
        # When receive_response() IS being consumed, the stop hook fires
        # mid-stream (before ResultMessage is yielded), so we must defer.
        async def stop_hook(input_data, tool_use_id, context):
            from mlflow.claude_code.hooks import get_hook_response

            if trace_built or receiving_response:
                return get_hook_response()

            session_id = input_data.get("session_id") if input_data else None
            _build_trace(messages_buffer, session_id)
            return get_hook_response()

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
