"""Autologging implementation for Pydantic AI 2.x."""

import functools
import inspect
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import is_dataclass
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.pydantic_ai.autolog_utils import parse_usage, serialize_output
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.provider import with_active_span
from mlflow.utils.autologging_utils import safe_patch
from mlflow.utils.autologging_utils.config import AutoLoggingConfig
from mlflow.utils.autologging_utils.safety import _store_patch, _wrap_patch

_logger = logging.getLogger(__name__)
_SAFE_PRIMITIVE_TYPES = (str, int, float, bool)


def _construct_inputs(func, *args, **kwargs) -> dict[str, Any]:
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs).arguments
        bound.pop("self", None)
        bound.pop("deps", None)
        return {
            key: (value.__dict__ if hasattr(value, "__dict__") else value)
            for key, value in bound.items()
            if value is not None
        }
    except (TypeError, ValueError):
        return kwargs


def _is_safe_for_serialization(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, _SAFE_PRIMITIVE_TYPES):
        return True
    if isinstance(value, dict):
        return all(_is_safe_for_serialization(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_safe_for_serialization(item) for item in value)
    return (is_dataclass(value) and not isinstance(value, type)) or isinstance(value, type)


def _extract_safe_attributes(instance: Any) -> dict[str, Any]:
    attributes = {}
    for key in dir(instance):
        if key.startswith("_"):
            continue
        try:
            value = getattr(instance, key, None)
        except Exception:
            continue
        if callable(value) and not isinstance(value, type):
            continue
        if isinstance(value, type):
            attributes[key] = value.__name__
        elif _is_safe_for_serialization(value):
            attributes[key] = value
    return attributes


def _set_agent_attributes(span: LiveSpan, agent) -> None:
    attributes = {SpanAttributeKey.MESSAGE_FORMAT: "pydantic_ai"}
    attributes.update(_extract_safe_attributes(agent))
    span.set_attributes(attributes)


def _set_model_attributes(span: LiveSpan, model) -> None:
    attributes = {SpanAttributeKey.MESSAGE_FORMAT: "pydantic_ai"}
    attributes.update(_extract_safe_attributes(model))
    span.set_attributes(attributes)

    if model_name := getattr(model, "model_name", None):
        span.set_attribute(SpanAttributeKey.MODEL, model_name)
        provider = getattr(model, "system", None)
        if provider is None and ":" in model_name:
            provider = model_name.split(":", 1)[0]
        if provider:
            span.set_attribute(SpanAttributeKey.MODEL_PROVIDER, provider)


def _set_result(span: LiveSpan, result: Any) -> None:
    span.set_outputs(serialize_output(result))
    if usage := parse_usage(result):
        span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)


def _model_request_inputs(request_context) -> dict[str, Any]:
    if request_context is None:
        return {}
    inputs = {
        "messages": getattr(request_context, "messages", None),
        "model_settings": getattr(request_context, "model_settings", None),
        "model_request_parameters": getattr(request_context, "model_request_parameters", None),
    }
    return {key: value for key, value in inputs.items() if value is not None}


def patched_agent_init(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if config.log_traces and self.instrument is None:
        self.instrument = True
    return result


async def patched_agent_run(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not config.log_traces:
        return await original(self, *args, **kwargs)

    with mlflow.start_span(name="Agent.run", span_type=SpanType.AGENT) as span:
        span.set_inputs(_construct_inputs(original, self, *args, **kwargs))
        _set_agent_attributes(span, self)
        result = await original(self, *args, **kwargs)
        _set_result(span, result)
        return result


def patched_agent_run_sync(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not config.log_traces:
        return original(self, *args, **kwargs)

    with mlflow.start_span(name="Agent.run_sync", span_type=SpanType.AGENT) as span:
        span.set_inputs(_construct_inputs(original, self, *args, **kwargs))
        _set_agent_attributes(span, self)
        result = original(self, *args, **kwargs)
        _set_result(span, result)
        return result


def patched_agent_run_stream(original, self, *args, **kwargs):
    @asynccontextmanager
    async def traced_stream():
        config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
        if not config.log_traces:
            async with original(self, *args, **kwargs) as result:
                yield result
            return

        with mlflow.start_span(name="Agent.run_stream", span_type=SpanType.AGENT) as span:
            span.set_inputs(_construct_inputs(original, self, *args, **kwargs))
            _set_agent_attributes(span, self)
            async with original(self, *args, **kwargs) as result:
                try:
                    yield result
                finally:
                    try:
                        _set_result(span, result)
                    except Exception as e:
                        _logger.debug("Failed to set streaming outputs: %s", e)

    return traced_stream()


async def patched_capability_model_request(original, self, ctx, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not config.log_traces:
        return await original(self, ctx, **kwargs)

    request_context = kwargs.get("request_context")
    model = getattr(request_context, "model", None)
    span_name = f"{type(model).__name__}.request" if model is not None else "Model.request"

    with mlflow.start_span(name=span_name, span_type=SpanType.LLM) as span:
        span.set_inputs(_model_request_inputs(request_context))
        if model is not None:
            _set_model_attributes(span, model)
        result = await original(self, ctx, **kwargs)
        _set_result(span, result)
        return result


async def patched_capability_tool_validate(
    original,
    self,
    ctx,
    *,
    call,
    tool_def,
    args,
    handler,
):
    """Trace validation only when it fails, leaving retry handling to Pydantic AI."""
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not config.log_traces:
        return await original(
            self,
            ctx,
            call=call,
            tool_def=tool_def,
            args=args,
            handler=handler,
        )

    from pydantic import ValidationError
    from pydantic_ai import ModelRetry

    try:
        return await original(
            self,
            ctx,
            call=call,
            tool_def=tool_def,
            args=args,
            handler=handler,
        )
    except (ValidationError, ModelRetry):
        with mlflow.start_span(
            name=f"{call.tool_name}.validation",
            span_type=SpanType.PARSER,
        ) as span:
            span.set_inputs(args)
            raise


async def patched_capability_tool_execute(
    original,
    self,
    ctx,
    *,
    call,
    tool_def,
    args,
    handler,
):
    """Trace a logical tool execution using only its public name, arguments, and result."""
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not config.log_traces:
        return await original(
            self,
            ctx,
            call=call,
            tool_def=tool_def,
            args=args,
            handler=handler,
        )

    with mlflow.start_span(name=call.tool_name, span_type=SpanType.TOOL) as span:
        span.set_inputs(args)
        result = await original(
            self,
            ctx,
            call=call,
            tool_def=tool_def,
            args=args,
            handler=handler,
        )
        span.set_outputs(serialize_output(result))
        return result


async def patched_mcp_list_tools(original, self):
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not config.log_traces:
        return await original(self)

    with mlflow.start_span(name="MCPToolset.list_tools", span_type=SpanType.TOOL) as span:
        span.set_inputs({})
        result = await original(self)
        span.set_outputs(serialize_output(result))
        return result


async def patched_mcp_direct_call_tool(
    original,
    self,
    name,
    args,
    *,
    metadata=None,
    use_task=False,
):
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not config.log_traces:
        return await original(
            self,
            name,
            args,
            metadata=metadata,
            use_task=use_task,
        )

    with mlflow.start_span(name="MCPToolset.direct_call_tool", span_type=SpanType.TOOL) as span:
        span.set_inputs({"name": name, "args": args})
        result = await original(
            self,
            name,
            args,
            metadata=metadata,
            use_task=use_task,
        )
        span.set_outputs(serialize_output(result))
        return result


class _StreamedRunResultSyncWrapper:
    """Keep a sync streaming span open until the Pydantic AI result is closed."""

    def __init__(self, result, span: LiveSpan):
        self._result = result
        self._span = span
        self._has_result_lifecycle = hasattr(result, "__exit__")
        self._closed = False
        self._finalized = False

    def _use_span_context(self):
        return with_active_span(self._span)

    def _close_result(self, exc_type=None, exc_value=None, traceback=None):
        if self._closed:
            return None
        self._closed = True
        if not self._has_result_lifecycle:
            return None
        with self._use_span_context():
            return self._result.__exit__(exc_type, exc_value, traceback)

    def _end_unfinished_child_spans(self) -> None:
        """Finish spans suspended by early Pydantic AI 2.x sync streaming.

        Early 2.x implemented ``run_stream_sync`` by advancing the async
        ``run_stream`` context manager to its first yield and did not expose a
        public close method. Consequently, its nested agent and model spans
        cannot exit naturally. Newer 2.x results implement ``__exit__`` and do
        not use this compatibility path.
        """
        if self._has_result_lifecycle:
            return

        from mlflow.tracing.trace_manager import InMemoryTraceManager

        manager = InMemoryTraceManager.get_instance()
        if manager is None:
            return

        with manager.get_trace(self._span.request_id) as trace:
            if trace is None:
                return

            unfinished = [
                span
                for span in trace.span_dict.values()
                if span.span_id != self._span.span_id and span.end_time_ns is None
            ]

            # End descendants before parents so the completed trace retains the
            # natural run_stream_sync -> run_stream -> model hierarchy.
            def depth(span):
                depth = 0
                parent_id = span.parent_id
                while parent_id and (parent := trace.span_dict.get(parent_id)):
                    depth += 1
                    parent_id = parent.parent_id
                return depth

            for child_span in sorted(unfinished, key=depth, reverse=True):
                child_span.end()

    def _end_span(self, exception: BaseException | None = None) -> None:
        try:
            _set_result(self._span, self._result)
        except Exception as e:
            _logger.debug("Failed to set streaming outputs: %s", e)

        if isinstance(exception, Exception):
            self._span.record_exception(exception)
        self._span.end(status="ERROR" if exception is not None else None)

    def _finalize(self, exc_type=None, exc_value=None, traceback=None):
        if self._finalized:
            return None
        self._finalized = True

        try:
            suppress = self._close_result(exc_type, exc_value, traceback)
            self._end_unfinished_child_spans()
        except BaseException as cleanup_error:
            self._end_span(cleanup_error)
            raise
        else:
            self._end_span(exc_value)
            return suppress

    def _wrap_iterator(self, iterator_func, **kwargs):
        try:
            with self._use_span_context():
                yield from iterator_func(**kwargs)
        except BaseException:
            self._finalize(*sys.exc_info())
            raise
        else:
            self._finalize()

    def stream_text(self, **kwargs):
        return self._wrap_iterator(self._result.stream_text, **kwargs)

    def stream_output(self, **kwargs):
        return self._wrap_iterator(self._result.stream_output, **kwargs)

    def stream_response(self, **kwargs):
        return self._wrap_iterator(self._result.stream_response, **kwargs)

    def get_output(self):
        try:
            with self._use_span_context():
                return self._result.get_output()
        except BaseException:
            self._finalize(*sys.exc_info())
            raise
        finally:
            if not self._finalized:
                self._finalize()

    def __enter__(self):
        if self._has_result_lifecycle:
            self._result.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._finalize(exc_type, exc_value, traceback)

    def __getattr__(self, name):
        return getattr(self._result, name)


def patched_agent_run_stream_sync(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not config.log_traces:
        return original(self, *args, **kwargs)

    span = mlflow.start_span_no_context(name="Agent.run_stream_sync", span_type=SpanType.AGENT)
    span.set_inputs(_construct_inputs(original, self, *args, **kwargs))
    _set_agent_attributes(span, self)

    try:
        with with_active_span(span):
            result = original(self, *args, **kwargs)
        return _StreamedRunResultSyncWrapper(result, span)
    except BaseException as e:
        if isinstance(e, Exception):
            span.record_exception(e)
        span.end(status="ERROR")
        raise


def _patch_streaming_method(cls, method_name, wrapper_func) -> None:
    original = getattr(cls, method_name)

    @functools.wraps(original)
    def patched_method(self, *args, **kwargs):
        return wrapper_func(original, self, *args, **kwargs)

    patch = _wrap_patch(cls, method_name, patched_method)
    _store_patch(mlflow.pydantic_ai.FLAVOR_NAME, patch)


def _patch_async_method(cls, method_name, wrapper_func) -> None:
    """Patch an async control-flow hook without marking handled exceptions as patch failures."""
    original = getattr(cls, method_name)

    @functools.wraps(original)
    async def patched_method(self, *args, **kwargs):
        return await wrapper_func(original, self, *args, **kwargs)

    patch = _wrap_patch(cls, method_name, patched_method)
    _store_patch(mlflow.pydantic_ai.FLAVOR_NAME, patch)


def _patch_agent_init(agent_cls) -> None:
    original = agent_cls.__init__

    @functools.wraps(original)
    def patched_init(self, *args, **kwargs):
        return patched_agent_init(original, self, *args, **kwargs)

    patch = _wrap_patch(agent_cls, "__init__", patched_init)
    _store_patch(mlflow.pydantic_ai.FLAVOR_NAME, patch)


def setup_autologging() -> None:
    """Install the Pydantic AI 2.x agent and model patches."""
    from pydantic_ai import Agent
    from pydantic_ai.capabilities.instrumentation import Instrumentation
    from pydantic_ai.mcp import MCPToolset

    _patch_agent_init(Agent)
    safe_patch(mlflow.pydantic_ai.FLAVOR_NAME, Agent, "run", patched_agent_run)
    safe_patch(mlflow.pydantic_ai.FLAVOR_NAME, Agent, "run_sync", patched_agent_run_sync)
    _patch_streaming_method(Agent, "run_stream", patched_agent_run_stream)
    _patch_streaming_method(Agent, "run_stream_sync", patched_agent_run_stream_sync)
    safe_patch(
        mlflow.pydantic_ai.FLAVOR_NAME,
        Instrumentation,
        "wrap_model_request",
        patched_capability_model_request,
    )
    # Validation and execution may raise ModelRetry as normal agent control flow. Using
    # safe_patch here would mark the enclosing autologging session as failed and suppress
    # instrumentation for the successful retry.
    _patch_async_method(
        Instrumentation,
        "wrap_tool_validate",
        patched_capability_tool_validate,
    )
    _patch_async_method(
        Instrumentation,
        "wrap_tool_execute",
        patched_capability_tool_execute,
    )
    _patch_async_method(MCPToolset, "list_tools", patched_mcp_list_tools)
    _patch_async_method(MCPToolset, "direct_call_tool", patched_mcp_direct_call_tool)
