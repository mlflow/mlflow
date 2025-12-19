import logging
from typing import Any

import mlflow
import mlflow.anthropic
from mlflow.anthropic.chat import convert_tool_to_mlflow_chat_tool
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.utils import (
    construct_full_inputs,
    set_span_chat_tools,
)
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def patched_claude_sdk_init(original, self, options=None):
    """Patched __init__ that adds MLflow tracing hook to ClaudeSDKClient.

    The hook handler checks autologging_is_disabled() at runtime, so hooks
    are always injected but become no-ops when autologging is disabled.
    """
    try:
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        from mlflow.claude_code.hooks import sdk_stop_hook_handler

        # Create options if not provided
        if options is None:
            options = ClaudeAgentOptions()

        if options.hooks is None:
            options.hooks = {}
        if "Stop" not in options.hooks:
            options.hooks["Stop"] = []

        options.hooks["Stop"].append(HookMatcher(hooks=[sdk_stop_hook_handler]))

        # Call original init with modified options
        return original(self, options)

    except Exception as e:
        _logger.debug("Error in patched_claude_sdk_init: %s", e, exc_info=True)
        # Fall back to original behavior if patching fails
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
