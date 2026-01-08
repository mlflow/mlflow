import contextvars
import inspect
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from typing import Any

from opentelemetry import trace

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)

# Context variable to track when we're inside run_stream_sync to prevent
# double span creation (run_stream_sync internally calls run_stream)
_in_sync_stream_context: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_in_sync_stream_context", default=False
)
_SAFE_PRIMITIVE_TYPES = (str, int, float, bool)


def _is_safe_for_serialization(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, _SAFE_PRIMITIVE_TYPES):
        return True
    if isinstance(value, dict):
        return all(_is_safe_for_serialization(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_safe_for_serialization(v) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        return True
    if isinstance(value, type):
        return True
    return False


def _safe_get_attribute(instance: Any, key: str) -> Any:
    try:
        value = getattr(instance, key, None)
        if value is None:
            return None
        if isinstance(value, type):
            return value.__name__
        if _is_safe_for_serialization(value):
            return value
        return None
    except Exception:
        return None


def _extract_safe_attributes(instance: Any) -> dict[str, Any]:
    """Extract all public attributes that are safe for serialization.

    Skips attributes starting with underscore to avoid capturing internal
    references (e.g., httpx clients) that can interfere with async cleanup.
    """
    attrs = {}
    for key in dir(instance):
        if key.startswith("_"):
            continue
        value = getattr(instance, key, None)
        # Skip methods/functions, but keep types (e.g., output_type=str)
        if callable(value) and not isinstance(value, type):
            continue
        safe_value = _safe_get_attribute(instance, key)
        if safe_value is not None:
            attrs[key] = safe_value
    return attrs


def _set_span_attributes(span: LiveSpan, instance):
    # 1) MCPServer attributes
    try:
        from pydantic_ai.mcp import MCPServer

        if isinstance(instance, MCPServer):
            mcp_attrs = _get_mcp_server_attributes(instance)
            span.set_attributes({k: v for k, v in mcp_attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving MCPServer attributes: %s", e)

    # 2) Agent attributes
    try:
        from pydantic_ai import Agent

        if isinstance(instance, Agent):
            agent_attrs = _get_agent_attributes(instance)
            span.set_attributes({k: v for k, v in agent_attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving Agent attributes: %s", e)

    # 3) InstrumentedModel attributes
    try:
        from pydantic_ai.models.instrumented import InstrumentedModel

        if isinstance(instance, InstrumentedModel):
            model_attrs = _get_model_attributes(instance)
            span.set_attributes({k: v for k, v in model_attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving InstrumentedModel attributes: %s", e)

    # 4) Tool attributes
    try:
        from pydantic_ai import Tool

        if isinstance(instance, Tool):
            tool_attrs = _get_tool_attributes(instance)
            span.set_attributes({k: v for k, v in tool_attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving Tool attributes: %s", e)


async def patched_async_class_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not cfg.log_traces:
        return await original(self, *args, **kwargs)

    fullname = f"{self.__class__.__name__}.{original.__name__}"
    span_type = _get_span_type(self)

    with mlflow.start_span(name=fullname, span_type=span_type) as span:
        inputs = _construct_full_inputs(original, self, *args, **kwargs)
        span.set_inputs(inputs)
        _set_span_attributes(span, self)

        result = await original(self, *args, **kwargs)
        outputs = _serialize_output(result)
        span.set_outputs(outputs)
        if usage_dict := _parse_usage(result):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
        return result


def patched_class_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not cfg.log_traces:
        return original(self, *args, **kwargs)

    fullname = f"{self.__class__.__name__}.{original.__name__}"
    span_type = _get_span_type(self)
    with mlflow.start_span(name=fullname, span_type=span_type) as span:
        inputs = _construct_full_inputs(original, self, *args, **kwargs)
        span.set_inputs(inputs)
        _set_span_attributes(span, self)

        result = original(self, *args, **kwargs)
        outputs = _serialize_output(result)
        span.set_outputs(outputs)
        if usage_dict := _parse_usage(result):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
        return result


def patched_async_stream_call(original, self, *args, **kwargs):
    @asynccontextmanager
    async def _wrapper():
        cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
        if not cfg.log_traces:
            async with original(self, *args, **kwargs) as result:
                yield result
            return

        # Skip span creation ONLY for Agent.run_stream when inside run_stream_sync.
        # Agent.run_stream_sync already creates a root span, so we don't need another
        # Agent.run_stream span. But we DO want InstrumentedModel spans (LLM calls).
        # The async context manager for Agent.run_stream won't properly exit when
        # called from run_stream_sync (pydantic_ai's implementation uses a generator
        # that pauses), so we skip it to avoid orphaned spans.
        from pydantic_ai import Agent

        if _in_sync_stream_context.get() and isinstance(self, Agent):
            async with original(self, *args, **kwargs) as result:
                yield result
            return

        fullname = f"{self.__class__.__name__}.{original.__name__}"
        span_type = _get_span_type(self)

        with mlflow.start_span(name=fullname, span_type=span_type) as span:
            inputs = _construct_full_inputs(original, self, *args, **kwargs)
            span.set_inputs(inputs)
            _set_span_attributes(span, self)

            async with original(self, *args, **kwargs) as stream_result:
                try:
                    yield stream_result
                finally:
                    # After the stream is consumed, get the final result
                    try:
                        outputs = _serialize_output(stream_result)
                        span.set_outputs(outputs)
                        if usage_dict := _parse_usage(stream_result):
                            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
                    except Exception as e:
                        _logger.debug(f"Failed to set streaming outputs: {e}")

    return _wrapper()


# Wrapper that captures span outputs after stream is consumed.
# This is necessary because run_stream_sync is NOT a context manager
# (unlike run_stream which is @asynccontextmanager). We must intercept
# iterator completion to know when streaming finishes.
class _StreamedRunResultSyncWrapper:
    def __init__(self, result, span):
        self._result = result
        self._span = span
        self._finalized = False

    def _use_span_context(self):
        return trace.use_span(self._span._span, end_on_exit=False)

    def _finalize(self):
        if self._finalized:
            return
        self._finalized = True

        # End child spans that haven't been ended yet.
        # This is necessary because pydantic_ai's run_stream_sync uses an async generator
        # that pauses mid-execution, causing async context managers (and their spans) to
        # never properly exit. We manually end these spans before ending the root span.
        self._end_unfinished_child_spans()

        try:
            self._span.set_outputs(_serialize_output(self._result))
            if usage_dict := _parse_usage(self._result):
                self._span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
        except Exception as e:
            _logger.debug(f"Failed to set streaming outputs: {e}")
        finally:
            self._span.end()

    def _end_unfinished_child_spans(self):
        from mlflow.tracing.trace_manager import InMemoryTraceManager

        manager = InMemoryTraceManager.get_instance()
        if manager is None:
            return

        trace_id = self._span.request_id
        root_span_id = self._span.span_id

        with manager.get_trace(trace_id) as trace:
            if not trace:
                return

            # Find and end all unfinished child spans (direct children of our root span)
            for span_id, span in trace.span_dict.items():
                if span_id == root_span_id:
                    continue
                # Only end spans that are direct children of our root span
                if span.parent_id == root_span_id and span._span.end_time is None:
                    try:
                        span.end()
                    except Exception as e:
                        _logger.debug(f"Failed to end child span {span.name}: {e}")

    def _wrap_iterator(self, iterator_func, **kwargs):
        with self._use_span_context():
            try:
                yield from iterator_func(**kwargs)
            finally:
                self._finalize()

    def stream_text(self, **kwargs):
        return self._wrap_iterator(self._result.stream_text, **kwargs)

    def stream_output(self, **kwargs):
        return self._wrap_iterator(self._result.stream_output, **kwargs)

    def stream_responses(self, **kwargs):
        return self._wrap_iterator(self._result.stream_responses, **kwargs)

    def get_output(self, **kwargs):
        with self._use_span_context():
            try:
                return self._result.get_output(**kwargs)
            finally:
                self._finalize()

    def __getattr__(self, name):
        return getattr(self._result, name)


def patched_sync_stream_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not cfg.log_traces:
        return original(self, *args, **kwargs)

    fullname = f"{self.__class__.__name__}.{original.__name__}"
    span_type = _get_span_type(self)

    # Use start_span_no_context (not `with start_span()`) because the span must remain
    # open after this function returns. The span ends later when the user finishes
    # iterating through the stream (handled by _StreamedRunResultSyncWrapper._finalize).
    span = mlflow.start_span_no_context(name=fullname, span_type=span_type)

    span.set_inputs(_construct_full_inputs(original, self, *args, **kwargs))
    _set_span_attributes(span, self)

    try:
        # Use use_span to set this span as the active context so child spans
        # (e.g., LLM calls via InstrumentedModel) are properly parented.
        # end_on_exit=False ensures we control when the span ends (in _finalize).
        # Also set _in_sync_stream_context to prevent patched_async_stream_call
        # from creating another Agent.run_stream span (it would never end due to
        # pydantic_ai's async generator implementation).
        token = _in_sync_stream_context.set(True)
        try:
            with trace.use_span(span._span, end_on_exit=False):
                result = original(self, *args, **kwargs)
        finally:
            _in_sync_stream_context.reset(token)

        return _StreamedRunResultSyncWrapper(result, span)
    except Exception:
        span.end(status="ERROR")
        raise


def _get_span_type(instance) -> str:
    try:
        from pydantic_ai import Agent, Tool
        from pydantic_ai.mcp import MCPServer
        from pydantic_ai.models.instrumented import InstrumentedModel
    except ImportError:
        return SpanType.UNKNOWN

    if isinstance(instance, InstrumentedModel):
        return SpanType.LLM
    if isinstance(instance, Agent):
        return SpanType.AGENT
    if isinstance(instance, Tool):
        return SpanType.TOOL
    if isinstance(instance, MCPServer):
        return SpanType.TOOL

    try:
        from pydantic_ai._tool_manager import ToolManager

        if isinstance(instance, ToolManager):
            return SpanType.TOOL
    except ImportError:
        pass
    return SpanType.UNKNOWN


def _construct_full_inputs(func, *args, **kwargs) -> dict[str, Any]:
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs).arguments
        bound.pop("self", None)
        bound.pop("deps", None)

        return {
            k: (v.__dict__ if hasattr(v, "__dict__") else v)
            for k, v in bound.items()
            if v is not None
        }
    except (ValueError, TypeError):
        return kwargs


def _serialize_output(result: Any) -> Any:
    if result is None:
        return None

    if hasattr(result, "new_messages") and callable(result.new_messages):
        try:
            new_messages = result.new_messages()
            serialized_messages = [asdict(msg) for msg in new_messages]

            try:
                serialized_result = asdict(result)
            except Exception:
                # We can't use asdict for StreamedRunResult because its async generator
                serialized_result = dict(result.__dict__) if hasattr(result, "__dict__") else {}

            serialized_result["_new_messages_serialized"] = serialized_messages
            return serialized_result
        except Exception as e:
            _logger.debug(f"Failed to serialize new_messages: {e}")

    return result.__dict__ if hasattr(result, "__dict__") else result


def _get_agent_attributes(instance):
    attrs = {SpanAttributeKey.MESSAGE_FORMAT: "pydantic_ai"}
    attrs.update(_extract_safe_attributes(instance))
    if hasattr(instance, "tools"):
        try:
            if tools_value := _parse_tools(instance.tools):
                attrs["tools"] = tools_value
        except Exception:
            pass
    return attrs


def _get_model_attributes(instance):
    attrs = {SpanAttributeKey.MESSAGE_FORMAT: "pydantic_ai"}
    attrs.update(_extract_safe_attributes(instance))
    return attrs


def _get_tool_attributes(instance):
    return _extract_safe_attributes(instance)


def _get_mcp_server_attributes(instance):
    attrs = _extract_safe_attributes(instance)
    if hasattr(instance, "tools"):
        try:
            if tools_value := _parse_tools(instance.tools):
                attrs["tools"] = tools_value
        except Exception:
            pass
    return attrs


def _parse_tools(tools):
    return [
        {"type": "function", "function": data}
        for tool in tools
        if (data := tool.model_dumps(exclude_none=True))
    ]


def _parse_usage(result: Any) -> dict[str, int] | None:
    try:
        if isinstance(result, tuple) and len(result) == 2:
            usage = result[1]
        else:
            usage_attr = getattr(result, "usage", None)
            if usage_attr is None:
                return None

            # Handle both property (RunResult) and method (StreamedRunResult)
            # StreamedRunResult has .usage() as a method
            usage = usage_attr() if callable(usage_attr) else usage_attr

        if usage is None:
            return None

        return {
            TokenUsageKey.INPUT_TOKENS: usage.request_tokens,
            TokenUsageKey.OUTPUT_TOKENS: usage.response_tokens,
            TokenUsageKey.TOTAL_TOKENS: usage.total_tokens,
        }
    except Exception as e:
        _logger.debug(f"Failed to parse token usage from output: {e}")
    return None
