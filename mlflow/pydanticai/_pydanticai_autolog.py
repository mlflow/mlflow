import atexit
import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Coroutine, Optional, TypeVar

import mlflow
import mlflow.tracking.fluent as _fluent
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    construct_full_inputs,
    end_client_span_or_trace,
    start_client_span_or_trace,
)
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)

FLAVOUR_NAME = "pydanticai"
_GLOBAL_SESSION_KEY = "__GLOBAL_SESSION__"
DEFAULT_SESSION_NAME = "PydanticAI"

_session_spans: dict[str, tuple[LiveSpan, Any]] = {}
_session_name = DEFAULT_SESSION_NAME


def _mlclient() -> mlflow.MlflowClient:
    return mlflow.MlflowClient()


def _normalize_result(result: Any) -> dict[str, Any]:
    """
    Turn any result (dict, dataclass, object) into a flat dict we can .get() on.
    """
    if isinstance(result, dict):
        return result
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "__dict__"):
        return vars(result)
    return {"value": result}


def _format_attribute_name(path: str) -> str:
    parts = path.split(".")
    return " ".join(p.lstrip("_").capitalize() for p in parts)


def _flatten_attributes(raw: dict[str, Any], exclude: Optional[dict[str, Any]]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def _recurse(obj: dict[str, Any], prefix: str = ""):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if k in exclude:
                continue
            if isinstance(v, dict):
                _recurse(v, full_key)
            else:
                name = _format_attribute_name(full_key)
                flat[name] = v

    _recurse(raw)
    return flat


def _capture_output_attributes(result, exclude: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Given the raw result, normalize it to a dict,
    pull out the "output" section, then flatten
    all of its nested fields into individual attrs.
    """
    raw = _normalize_result(result)
    output_dict = raw.get("output", raw)
    return _flatten_attributes(output_dict, exclude=exclude)


def _start_span(
    *,
    name: str,
    span_type: SpanType | str,
    inputs: dict[str, Any],
    attributes: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> LiveSpan:
    """Create a client span, optionally linking it to `run_id`."""
    span = start_client_span_or_trace(
        _mlclient(), name=name, span_type=span_type, inputs=inputs, attributes=attributes
    )
    if run_id:
        InMemoryTraceManager().get_instance().set_request_metadata(
            span.request_id, TraceMetadataKey.SOURCE_RUN, run_id
        )
    return span


def _end_span_ok(span: LiveSpan, outputs: Any, attributes: Optional[dict[str, Any]] = None) -> None:
    try:
        end_client_span_or_trace(_mlclient(), span, outputs=outputs, attributes=attributes)
    except Exception as exc:
        _logger.debug("Failed to end span: %s", exc, exc_info=True)


def _end_span_err(span: LiveSpan, exc: BaseException) -> None:
    span.add_event(SpanEvent.from_exception(exc))
    _mlclient().end_span(span.request_id, span.span_id, status=SpanStatusCode.ERROR)


T = TypeVar("T")


def _with_span_async(
    *,
    span_name: str | Callable[[Any], str],
    span_type: SpanType | str,
    capture_inputs: Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]],
    capture_input_attributes: Callable[[Any], dict[str, Any]] = lambda args, kwargs: {},
) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Returns a function of signature (original, *args, **kwargs) -> Coroutine.
    """

    async def _wrapper(
        original: Callable[..., Coroutine[Any, Any, T]],
        *call_args: Any,
        **call_kwargs: Any,
    ) -> T:
        self_obj, *user_args = call_args
        cfg = AutoLoggingConfig.init(flavor_name=FLAVOUR_NAME)
        if not (cfg and cfg.log_traces):
            return await original(self_obj, *user_args, **call_kwargs)

        # ensure a single session span groups all calls
        _get_or_create_session_span()

        active_run = mlflow.active_run()
        run_id = active_run.info.run_id if active_run else None

        name = span_name(self_obj) if callable(span_name) else span_name
        # build inputs
        inputs = capture_inputs(call_args, call_kwargs)
        attributes = capture_input_attributes(call_args, call_kwargs)
        span = _start_span(
            name=name, span_type=span_type, inputs=inputs, run_id=run_id, attributes=attributes
        )
        token = set_span_in_context(span)

        try:
            result = await original(self_obj, *user_args, **call_kwargs)
        except Exception as exc:
            _end_span_err(span, exc)
            raise
        finally:
            detach_span_from_context(token)

        output = _normalize_result(result).get("output", result)
        output_attributes = _capture_output_attributes(output, exclude={"output"})
        _end_span_ok(span, outputs={"output": output}, attributes=output_attributes)
        return result

    return _wrapper


def _with_span_sync(
    *,
    span_name: str | Callable[[Any], str],
    span_type: SpanType | str,
    capture_inputs: Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]],
    capture_input_attributes: Callable[[Any], dict[str, Any]] = lambda args, kwargs: {},
) -> Callable[..., T]:
    """
    Returns a function of signature (original, *args, **kwargs) -> T.
    """

    def _wrapper(
        original: Callable[..., T],
        *call_args: Any,
        **call_kwargs: Any,
    ) -> T:
        self_obj, *user_args = call_args
        cfg = AutoLoggingConfig.init(flavor_name=FLAVOUR_NAME)
        if not (cfg and cfg.log_traces):
            return original(self_obj, *user_args, **call_kwargs)

        _get_or_create_session_span()

        active_run = mlflow.active_run()
        run_id = active_run.info.run_id if active_run else None
        name = span_name(self_obj) if callable(span_name) else span_name
        inputs = capture_inputs(call_args, call_kwargs)
        attributes = capture_input_attributes(call_args, call_kwargs)

        span = _start_span(
            name=name, span_type=span_type, inputs=inputs, run_id=run_id, attributes=attributes
        )
        token = set_span_in_context(span)

        try:
            result = original(self_obj, *user_args, **call_kwargs)
        except Exception as exc:
            _end_span_err(span, exc)
            raise
        finally:
            detach_span_from_context(token)

        output = _normalize_result(result).get("output", result)
        output_attributes = _capture_output_attributes(output, exclude={"output"})
        _end_span_ok(span, outputs={"output": output}, attributes=output_attributes)
        return result

    return _wrapper


def _get_or_create_session_span() -> None:
    """
    Ensure a *single* session-level chain span exists for the current process
    (or for the active MLflow run).  This groups all nested agent calls so
    they appear under one trace in the UI.
    """
    run = mlflow.active_run()
    session_key = run.info.run_id if run else _GLOBAL_SESSION_KEY
    if session_key in _session_spans:
        return

    span = start_client_span_or_trace(
        _mlclient(), name=_session_name, span_type=SpanType.CHAIN, inputs={}
    )
    token = set_span_in_context(span)
    _session_spans[session_key] = (span, token)

    if run:
        InMemoryTraceManager().get_instance().set_request_metadata(
            span.request_id, TraceMetadataKey.SOURCE_RUN, run.info.run_id
        )

    else:

        def _close_global_session() -> None:
            _span, _tok = _session_spans.pop(_GLOBAL_SESSION_KEY, (None, None))
            if _tok:
                detach_span_from_context(_tok)
            if _span:
                end_client_span_or_trace(_mlclient(), _span, outputs=None)

        atexit.register(_close_global_session)


def _patched_end_run(original_end_run, status: str | None = None, *args, **kwargs):
    """
    Finish the session span *before* MLflow ends the run so the UI renders
    everything hierarchically.
    """
    run = mlflow.active_run()
    if run and run.info.run_id in _session_spans:
        span, token = _session_spans.pop(run.info.run_id)
        detach_span_from_context(token)
        end_client_span_or_trace(_mlclient(), span, outputs=None)
    return original_end_run(status, *args, **kwargs)


def _capture_inputs(original: Callable, self_obj: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Helper passed into decorators so we *consistently* derive span.inputs."""
    return construct_full_inputs(original, self_obj, *args, **kwargs)


def _patch_agent_methods() -> None:
    import pydantic_ai

    safe_patch(
        FLAVOUR_NAME,
        pydantic_ai.Agent,
        "run_sync",
        _with_span_sync(
            span_name=lambda s: f"{s.__class__.__name__}.run_sync",
            span_type=SpanType.CHAIN,
            capture_inputs=lambda args, kwargs: {"message": args[1]} if len(args) > 1 else {},
            capture_input_attributes=lambda args, kwargs: kwargs,
        ),
    )
    safe_patch(
        FLAVOUR_NAME,
        pydantic_ai.Agent,
        "run",
        _with_span_async(
            span_name=lambda s: f"{s.__class__.__name__}.run",
            span_type=SpanType.CHAIN,
            capture_inputs=lambda args, kwargs: {"message": args[1]} if len(args) > 1 else {},
            capture_input_attributes=lambda args, kwargs: kwargs,
        ),
    )


def _patch_instrumented_model() -> None:
    from pydantic_ai.models.instrumented import InstrumentedModel

    safe_patch(
        FLAVOUR_NAME,
        InstrumentedModel,
        "request",
        _with_span_async(
            span_name=lambda s: (
                f"""{
                    getattr(
                        getattr(s, "wrapped", s),
                        "provider_name",
                        getattr(s, "wrapped", s).__class__.__name__,
                    )
                }.request"""
            ),
            span_type=SpanType.LLM,
            capture_inputs=lambda args, kwargs: {"message": args[1]} if len(args) > 1 else {},
            capture_input_attributes=lambda args, kwargs: dict(kwargs),
        ),
    )


def _get_tool_run_attributes(
    self_obj: Any, message: Any, run_context: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = {
        "tool_name": self_obj.name,
        "tool_call_id": message.tool_call_id,
        "tool_arguments": json.loads(message.args_as_json_str()),
    }

    attributes = {
        "model_class": run_context.model.__class__.__name__,
        "model_name": getattr(run_context.model, "model_name", None),
        "prompt": run_context.prompt,
        "messages": [asdict(m) for m in run_context.messages],
        "usage": {
            "request_tokens": run_context.usage.request_tokens,
            "response_tokens": run_context.usage.response_tokens,
            "total_tokens": run_context.usage.total_tokens,
            **(
                {"details": run_context.usage.details}
                if getattr(run_context.usage, "details", None) is not None
                else {}
            ),
        },
        "retry": run_context.retry,
        "run_step": run_context.run_step,
    }

    return inputs, attributes


def _patch_tool_run() -> None:
    from pydantic_ai.tools import Tool

    async def _tool_run_wrapper(original, self_obj, message, run_context, tracer):
        cfg = AutoLoggingConfig.init(flavor_name=FLAVOUR_NAME)
        if not (cfg and cfg.log_traces):
            return await original(self_obj, message, run_context, tracer)

        inputs, attributes = _get_tool_run_attributes(self_obj, message, run_context)

        span = _start_span(
            name=self_obj.name, span_type=SpanType.TOOL, inputs=inputs, attributes=attributes
        )
        try:
            result = await original(self_obj, message, run_context, tracer)
        except Exception as exc:
            _end_span_err(span, exc)
            raise

        output = _normalize_result(result).get("output", result)
        output_attributes = _capture_output_attributes(output, exclude={"output"})
        _end_span_ok(span, outputs={"output": output}, attributes=output_attributes)
        return result

    safe_patch(FLAVOUR_NAME, Tool, "run", _tool_run_wrapper)


def _patch_mcp_server() -> None:
    from pydantic_ai.mcp import MCPServer

    _mcp_call_tool_decorator = _with_span_async(
        span_name=lambda self: f"{self.__class__.__name__}.call_tool",
        span_type=SpanType.CHAIN,
        capture_inputs=lambda args, kwargs: {
            "tool_name": args[1],
        },
        capture_input_attributes=lambda args, kwargs: {
            "tool_arguments": args[2],
        },
    )

    async def _call_tool_patch(original, *args, **kwargs):
        _get_or_create_session_span()
        return await _mcp_call_tool_decorator(original, *args, **kwargs)

    safe_patch(FLAVOUR_NAME, MCPServer, "call_tool", _call_tool_patch)

    _mcp_list_tools_decorator = _with_span_async(
        span_name=lambda self: f"{self.__class__.__name__}.list_tools",
        span_type=SpanType.CHAIN,
        capture_inputs=lambda args, kwargs: {},
        capture_input_attributes=lambda args, kwargs: {},
    )

    async def _list_tools_patch(original, *args, **kwargs):
        _get_or_create_session_span()
        return await _mcp_list_tools_decorator(original, *args, **kwargs)

    safe_patch(FLAVOUR_NAME, MCPServer, "list_tools", _list_tools_patch)


@autologging_integration(FLAVOUR_NAME)
def autolog(
    *,
    log_traces: bool = True,
    extra_tags: dict[str, str] | None = None,
    disable: bool = False,
    silent: bool = False,
    session_name: str = DEFAULT_SESSION_NAME,
) -> None:
    """
    Enable Pydantic-AI auto-instrumentation for MLflow.

    Parameters
    ----------
    log_traces
        Toggle collection of MLflow traces/spans.
    extra_tags
        (Reserved for future use) – extra tags to attach to every span.
    disable
        Bypass instrumentation entirely (convenience flag).
    silent
        Currently unused – kept for parity with other MLflow flavours.
    session_name
        Human-readable name for the root *session* span.
    """
    if disable:
        _logger.info("pydantic-ai autologging disabled by caller")
        return

    global _session_name
    _session_name = session_name

    # store config once so downstream helpers can access it quickly
    AutoLoggingConfig.init(flavor_name=FLAVOUR_NAME)

    safe_patch(FLAVOUR_NAME, _fluent, "end_run", _patched_end_run)
    safe_patch(FLAVOUR_NAME, mlflow, "end_run", _patched_end_run)

    _patch_agent_methods()
    _patch_instrumented_model()
    _patch_tool_run()
    _patch_mcp_server()

    _logger.debug("Pydantic-AI autologging enabled (traces=%s)", log_traces)
