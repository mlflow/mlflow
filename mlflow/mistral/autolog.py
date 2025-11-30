import inspect
import logging

import mlflow
import mlflow.mistral
from mlflow.entities import SpanType
from mlflow.mistral.chat import convert_tool_to_mlflow_chat_tool
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context
from mlflow.tracing.utils import set_span_chat_tools
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def _construct_full_inputs(func, *args, **kwargs):
    signature = inspect.signature(func)
    # this does not create copy. So values should not be mutated directly
    arguments = signature.bind_partial(*args, **kwargs).arguments

    if "self" in arguments:
        arguments.pop("self")

    return arguments


def patched_class_call(original, self, *args, **kwargs):
    """Synchronous wrapper that traces Mistral SDK calls using a context manager."""
    with TracingSession(original, self, args, kwargs) as manager:
        output = original(self, *args, **kwargs)
        manager.output = output
        return output


async def async_patched_class_call(original, self, *args, **kwargs):
    """Async wrapper that traces Mistral SDK calls using a context manager."""
    async with TracingSession(original, self, args, kwargs) as manager:
        output = await original(self, *args, **kwargs)
        manager.output = output
        return output


class TracingSession:
    """Context manager for handling MLflow spans in both sync and async contexts."""

    def __init__(self, original, instance, args, kwargs):
        self.original = original
        self.instance = instance
        self.inputs = _construct_full_inputs(original, instance, *args, **kwargs)

        # These attributes are set outside the constructor.
        self.span = None
        self.token = None
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
        config = AutoLoggingConfig.init(flavor_name=mlflow.mistral.FLAVOR_NAME)
        if not config.log_traces:
            return self

        self.span = mlflow.start_span_no_context(
            name=f"{self.instance.__class__.__name__}.{self.original.__name__}",
            span_type=SpanType.CHAT_MODEL,
            inputs=self.inputs,
            attributes={SpanAttributeKey.MESSAGE_FORMAT: "mistral"},
        )

        if (tools := self.inputs.get("tools")) is not None:
            try:
                tools = [convert_tool_to_mlflow_chat_tool(tool) for tool in tools if tool]
                set_span_chat_tools(self.span, tools)
            except Exception as e:
                _logger.debug(f"Failed to set tools for {self.span}. Error: {e}")

        # Attach the span to the current context. A single SDK call can create child spans.
        self.token = set_span_in_context(self.span)
        return self

    def _exit_impl(self, exc_type, exc_val, exc_tb) -> None:
        if not self.span:
            return

        # Detach span from the context first to avoid leaking the context on errors.
        detach_span_from_context(self.token)

        if exc_val:
            self.span.record_exception(exc_val)

        try:
            if usage := _parse_usage(self.output):
                self.span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)
        except Exception as e:
            _logger.debug(
                f"Failed to extract token usage for span {self.span.name}: {e}",
                exc_info=True,
            )

        # End the span with captured outputs. Keep original object for backward compatibility.
        self.span.end(outputs=self.output)


def _parse_usage(output):
    usage = getattr(output, "usage", None)
    if usage is None:
        return None

    usage_dict = {}
    if getattr(usage, "prompt_tokens", None) is not None:
        usage_dict[TokenUsageKey.INPUT_TOKENS] = usage.prompt_tokens
    if getattr(usage, "completion_tokens", None) is not None:
        usage_dict[TokenUsageKey.OUTPUT_TOKENS] = usage.completion_tokens
    if getattr(usage, "total_tokens", None) is not None:
        usage_dict[TokenUsageKey.TOTAL_TOKENS] = usage.total_tokens

    return usage_dict or None
