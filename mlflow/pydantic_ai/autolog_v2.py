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
from mlflow.utils.autologging_utils import get_autologging_config, is_testing, safe_patch
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


async def patched_capability_tool_validate_error(
    original,
    self,
    ctx,
    *,
    call,
    tool_def,
    args,
    error,
    **kwargs,
):
    """Trace failed validation while leaving error handling to Pydantic AI."""
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not config.log_traces:
        return await original(
            self,
            ctx,
            call=call,
            tool_def=tool_def,
            args=args,
            error=error,
            **kwargs,
        )

    with mlflow.start_span(
        name=f"{call.tool_name}.validation",
        span_type=SpanType.PARSER,
    ) as span:
        span.set_inputs(args)
        return await original(
            self,
            ctx,
            call=call,
            tool_def=tool_def,
            args=args,
            error=error,
            **kwargs,
        )


async def patched_capability_tool_execute(
    original,
    self,
    ctx,
    *,
    call,
    tool_def,
    args,
    handler,
    **kwargs,
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
            **kwargs,
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
            **kwargs,
        )
        span.set_outputs(serialize_output(result))
        return result


async def patched_mcp_list_tools(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not config.log_traces:
        return await original(self, *args, **kwargs)

    with mlflow.start_span(name="MCPToolset.list_tools", span_type=SpanType.TOOL) as span:
        span.set_inputs({})
        result = await original(self, *args, **kwargs)
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
    **kwargs,
):
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not config.log_traces:
        return await original(
            self,
            name,
            args,
            metadata=metadata,
            use_task=use_task,
            **kwargs,
        )

    with mlflow.start_span(name="MCPToolset.direct_call_tool", span_type=SpanType.TOOL) as span:
        span.set_inputs({"name": name, "args": args})
        result = await original(
            self,
            name,
            args,
            metadata=metadata,
            use_task=use_task,
            **kwargs,
        )
        span.set_outputs(serialize_output(result))
        return result


class _StreamedRunResultSyncWrapper:
    """Keep a sync streaming span open until the Pydantic AI result is closed."""

    def __init__(self, result, span: LiveSpan):
        self._result = result
        self._span = span
        self._closed = False
        self._finalized = False

    def _use_span_context(self):
        return with_active_span(self._span)

    def _close_result(self, exc_type=None, exc_value=None, traceback=None):
        if self._closed:
            return None
        self._closed = True
        with self._use_span_context():
            return self._result.__exit__(exc_type, exc_value, traceback)

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
    """Patch an async control-flow hook without treating library exceptions as patch failures."""
    original = getattr(cls, method_name)

    @functools.wraps(original)
    async def patched_method(self, *args, **kwargs):
        original_result = None
        original_succeeded = False
        original_exception = None
        original_traceback = None

        async def call_original(*original_args, **original_kwargs):
            nonlocal original_exception
            nonlocal original_result
            nonlocal original_succeeded
            nonlocal original_traceback

            try:
                original_result = await original(*original_args, **original_kwargs)
                original_succeeded = True
                return original_result
            except BaseException as e:
                original_exception = e
                original_traceback = e.__traceback__
                raise

        try:
            return await wrapper_func(call_original, self, *args, **kwargs)
        except BaseException as patch_error:
            if original_exception is not None:
                raise original_exception.with_traceback(original_traceback)
            if not isinstance(patch_error, Exception) or is_testing():
                raise

            if not get_autologging_config(
                mlflow.pydantic_ai.FLAVOR_NAME,
                "silent",
                False,
            ):
                _logger.warning(
                    "Encountered unexpected error during Pydantic AI autologging: %s",
                    patch_error,
                )

            # The original call may already have produced a result before an MLflow postamble
            # failed. Return it rather than executing a tool or transport operation twice.
            if original_succeeded:
                return original_result
            return await original(self, *args, **kwargs)

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
    _patch_async_method(
        Instrumentation,
        "on_tool_validate_error",
        patched_capability_tool_validate_error,
    )
    _patch_async_method(
        Instrumentation,
        "wrap_tool_execute",
        patched_capability_tool_execute,
    )
    safe_patch(
        mlflow.pydantic_ai.FLAVOR_NAME,
        MCPToolset,
        "list_tools",
        patched_mcp_list_tools,
    )
    _patch_async_method(
        MCPToolset,
        "direct_call_tool",
        patched_mcp_direct_call_tool,
    )
