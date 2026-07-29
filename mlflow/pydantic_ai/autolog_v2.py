import functools
import inspect
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.pydantic_ai.utils import (
    extract_safe_attributes,
    model_request_inputs,
    parse_usage,
    serialize_output,
)
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.provider import with_active_span
from mlflow.utils import autologging_utils
from mlflow.utils.autologging_utils import (
    autologging_is_disabled,
    get_autologging_config,
    is_testing,
    safe_patch,
)
from mlflow.utils.autologging_utils.config import AutoLoggingConfig
from mlflow.utils.autologging_utils.safety import _store_patch, _wrap_patch

_logger = logging.getLogger(__name__)


def _tracing_enabled() -> bool:
    # The wrappers installed by _patch_streaming_method and _safe_patch_async_hook bypass
    # safe_patch, so they must replicate its gating: honor the per-flavor log_traces config,
    # the per-flavor disabled state, AND the process-wide disable_autologging() flag. The
    # latter is a separate module global that autologging_is_disabled() does not consult, so
    # it must be checked explicitly — otherwise these manual patches keep emitting spans
    # under global suppression, leaking inputs such as tool arguments.
    if autologging_utils._AUTOLOGGING_GLOBALLY_DISABLED:
        return False
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    return config.log_traces and not autologging_is_disabled(mlflow.pydantic_ai.FLAVOR_NAME)


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


def _set_agent_attributes(span: LiveSpan, agent) -> None:
    attributes = {SpanAttributeKey.MESSAGE_FORMAT: "pydantic_ai"}
    attributes.update(extract_safe_attributes(agent))
    span.set_attributes(attributes)


def _set_model_attributes(span: LiveSpan, model) -> None:
    attributes = {SpanAttributeKey.MESSAGE_FORMAT: "pydantic_ai"}
    attributes.update(extract_safe_attributes(model))
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


def patched_agent_init(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    config = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if config.log_traces and getattr(self, "instrument", None) is None:
        self.instrument = True
    return result


async def patched_agent_run(original, self, *args, **kwargs):
    if not _tracing_enabled():
        return await original(self, *args, **kwargs)

    with mlflow.start_span(name="Agent.run", span_type=SpanType.AGENT) as span:
        span.set_inputs(_construct_inputs(original, self, *args, **kwargs))
        _set_agent_attributes(span, self)
        result = await original(self, *args, **kwargs)
        _set_result(span, result)
        return result


def patched_agent_run_sync(original, self, *args, **kwargs):
    if not _tracing_enabled():
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
        if not _tracing_enabled():
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
    if not _tracing_enabled():
        return await original(self, ctx, **kwargs)

    request_context = kwargs.get("request_context")
    model = getattr(request_context, "model", None)
    span_name = f"{type(model).__name__}.request" if model is not None else "Model.request"

    with mlflow.start_span(name=span_name, span_type=SpanType.LLM) as span:
        span.set_inputs(model_request_inputs(request_context))
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
    if not _tracing_enabled():
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
    def __init__(self, result, span: LiveSpan):
        self._result = result
        self._span = span
        self._closed = False
        self._finalized = False
        self._entered = False

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

    # While the caller holds the result open in a `with` block, __exit__ owns finalization.
    # Closing here would exit the underlying Pydantic AI result before the caller is done with
    # it, breaking follow-up calls such as get_output() after a stream has been consumed.
    def _finalize_unless_entered(self, exc_type=None, exc_value=None, traceback=None):
        if self._entered:
            return None
        return self._finalize(exc_type, exc_value, traceback)

    def _wrap_iterator(self, iterator_func, **kwargs):
        try:
            with self._use_span_context():
                yield from iterator_func(**kwargs)
        except BaseException:
            self._finalize_unless_entered(*sys.exc_info())
            raise
        else:
            self._finalize_unless_entered()

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
            self._finalize_unless_entered(*sys.exc_info())
            raise
        finally:
            if not self._finalized:
                self._finalize_unless_entered()

    def __enter__(self):
        self._result.__enter__()
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._finalize(exc_type, exc_value, traceback)

    def __getattr__(self, name):
        return getattr(self._result, name)


def patched_agent_run_stream_sync(original, self, *args, **kwargs):
    if not _tracing_enabled():
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


# Pydantic AI uses exceptions such as ModelRetry for control flow. General safe_patch would
# mark the shared autologging session as failed and suppress tracing for the successful retry.
def _safe_patch_async_hook(destination, function_name, patch_function) -> None:
    original = getattr(destination, function_name)

    @functools.wraps(original)
    async def patched_method(self, *args, **kwargs):
        original_has_been_called = False
        original_result = None
        failed_during_original = False

        @functools.wraps(original)
        async def call_original(*original_args, **original_kwargs):
            nonlocal failed_during_original
            nonlocal original_has_been_called
            nonlocal original_result

            original_has_been_called = True
            try:
                original_result = await original(*original_args, **original_kwargs)
                return original_result
            except BaseException:
                failed_during_original = True
                raise

        try:
            if not _tracing_enabled():
                return await call_original(self, *args, **kwargs)
            return await patch_function(call_original, self, *args, **kwargs)
        except BaseException as patch_error:
            if failed_during_original:
                raise
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
            if original_has_been_called:
                return original_result
            return await original(self, *args, **kwargs)

    patch = _wrap_patch(destination, function_name, patched_method)
    _store_patch(mlflow.pydantic_ai.FLAVOR_NAME, patch)


def setup_autologging() -> None:
    from pydantic_ai import Agent

    safe_patch(
        mlflow.pydantic_ai.FLAVOR_NAME,
        Agent,
        "__init__",
        patched_agent_init,
    )
    safe_patch(mlflow.pydantic_ai.FLAVOR_NAME, Agent, "run", patched_agent_run)
    safe_patch(mlflow.pydantic_ai.FLAVOR_NAME, Agent, "run_sync", patched_agent_run_sync)
    _patch_streaming_method(Agent, "run_stream", patched_agent_run_stream)
    _patch_streaming_method(Agent, "run_stream_sync", patched_agent_run_stream_sync)

    # Instrumentation and MCP are optional surfaces (e.g. MCPToolset only exists when the
    # `mcp` extra is installed). Degrade gracefully like the 1.x path so a missing optional
    # module disables only that surface instead of failing the whole autolog() call.
    try:
        from pydantic_ai.capabilities.instrumentation import Instrumentation

        safe_patch(
            mlflow.pydantic_ai.FLAVOR_NAME,
            Instrumentation,
            "wrap_model_request",
            patched_capability_model_request,
        )
        _safe_patch_async_hook(
            Instrumentation,
            "on_tool_validate_error",
            patched_capability_tool_validate_error,
        )
        _safe_patch_async_hook(
            Instrumentation,
            "wrap_tool_execute",
            patched_capability_tool_execute,
        )
    except (ImportError, AttributeError) as e:
        _logger.warning("Skipping Pydantic AI 2.x Instrumentation tracing: %s", e)

    try:
        from pydantic_ai.mcp import MCPToolset

        safe_patch(
            mlflow.pydantic_ai.FLAVOR_NAME,
            MCPToolset,
            "list_tools",
            patched_mcp_list_tools,
        )
        _safe_patch_async_hook(
            MCPToolset,
            "direct_call_tool",
            patched_mcp_direct_call_tool,
        )
    except (ImportError, AttributeError) as e:
        _logger.warning("Skipping Pydantic AI 2.x MCP tracing: %s", e)
