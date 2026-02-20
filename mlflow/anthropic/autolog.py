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
    try:
        from claude_agent_sdk.types import UserMessage

        result = original(self, options)
        messages = []

        # query() sends the user prompt but doesn't echo it through receive_response()
        original_query = self.query

        async def wrapped_query(prompt, *args, **kwargs):
            if isinstance(prompt, str):
                messages.append(UserMessage(content=prompt))
            elif hasattr(prompt, "__aiter__"):
                # prompt is an async generator yielding message dicts — wrap it
                # to capture the user content while passing items through to the SDK
                original_prompt = prompt

                async def capturing_prompt():
                    async for item in original_prompt:
                        if isinstance(item, dict) and item.get("type") == "user":
                            content = item.get("message", {}).get("content", "")
                            if isinstance(content, str) and content.strip():
                                messages.append(UserMessage(content=content))
                        yield item

                prompt = capturing_prompt()
            return await original_query(prompt, *args, **kwargs)

        self.query = wrapped_query

        original_receive_response = self.receive_response

        async def wrapped_receive_response(*args, **kwargs):
            async for msg in original_receive_response(*args, **kwargs):
                messages.append(msg)
                yield msg
            try:
                from mlflow.utils.autologging_utils import autologging_is_disabled

                if not autologging_is_disabled("anthropic"):
                    from mlflow.claude_code.tracing import process_sdk_messages

                    process_sdk_messages(list(messages))
            except Exception as e:
                _logger.debug("Error building SDK trace: %s", e, exc_info=True)

        self.receive_response = wrapped_receive_response
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
            usage_dict = {
                TokenUsageKey.INPUT_TOKENS: usage.input_tokens,
                TokenUsageKey.OUTPUT_TOKENS: usage.output_tokens,
                TokenUsageKey.TOTAL_TOKENS: usage.input_tokens + usage.output_tokens,
            }
            if (cached := getattr(usage, "cache_read_input_tokens", None)) is not None:
                usage_dict[TokenUsageKey.CACHE_READ_INPUT_TOKENS] = cached
            if (created := getattr(usage, "cache_creation_input_tokens", None)) is not None:
                usage_dict[TokenUsageKey.CACHE_CREATION_INPUT_TOKENS] = created
            return usage_dict
    except Exception as e:
        _logger.debug(f"Failed to parse token usage from output: {e}")
    return None
