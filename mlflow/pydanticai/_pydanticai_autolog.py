from __future__ import annotations

import atexit
import json
import logging
from copy import deepcopy
from dataclasses import asdict
from typing import Any, AsyncIterator, Optional

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

_session_spans: dict[str, tuple[Any, Any]] = {}

_logger = logging.getLogger(__name__)
FLAVOUR_NAME: str = "pydanticai"


def _patched_end_run(original, *args, **kwargs):
    run = mlflow.active_run()
    if run and run.info.run_id in _session_spans:
        span, token = _session_spans.pop(run.info.run_id)
        detach_span_from_context(token)
        end_client_span_or_trace(mlflow.MlflowClient(), span, outputs=None)

    status_args = args[:1]
    return original(*status_args, **kwargs)


# TODO: Avoid creating new runs for each tracing
def _get_or_create_session_span():
    run = mlflow.active_run()
    if run is None:
        mlflow.start_run()
        atexit.register(mlflow.end_run)
        run = mlflow.active_run()
    run_id = run.info.run_id
    if run_id not in _session_spans:
        client = mlflow.MlflowClient()
        span = start_client_span_or_trace(
            client, name="PydanticAI Session", span_type=SpanType.CHAIN, inputs={}
        )
        token = set_span_in_context(span)
        _session_spans[run_id] = (span, token)


def _start_span(
    mlclient: mlflow.MlflowClient,
    name: str,
    span_type: str,
    inputs: dict[str, Any],
    run_id: Optional[str],
) -> LiveSpan:
    span = start_client_span_or_trace(mlclient, name=name, span_type=span_type, inputs=inputs)
    if run_id:
        InMemoryTraceManager().get_instance().set_request_metadata(
            span.request_id, TraceMetadataKey.SOURCE_RUN, run_id
        )
    return span


def _end_span_ok(mlclient: mlflow.MlflowClient, span: LiveSpan, outputs: Any) -> None:
    try:
        end_client_span_or_trace(mlclient, span, outputs=outputs)
    except Exception as e:
        _logger.debug("ending span failed: %s", e, exc_info=True)


def _end_span_err(mlclient: mlflow.MlflowClient, span: LiveSpan, exc: BaseException) -> None:
    span.add_event(SpanEvent.from_exception(exc))
    mlclient.end_span(span.request_id, span.span_id, status=SpanStatusCode.ERROR)


async def _patched_run(original: Any, self_obj: Any, *args: Any, **kwargs: Any) -> Any:
    _get_or_create_session_span()

    cfg = AutoLoggingConfig.init(flavor_name=FLAVOUR_NAME)
    mlclient = mlflow.MlflowClient()

    inputs = construct_full_inputs(original, self_obj, *args, **kwargs)

    span = None
    token = None
    if cfg.log_traces:
        span = _start_span(
            mlclient,
            name=f"{self_obj.__class__.__name__}.run",
            span_type=SpanType.CHAIN,
            inputs=inputs,
            run_id=mlflow.active_run().info.run_id if mlflow.active_run() else None,
        )
        token = set_span_in_context(span)

    try:
        result = await original(self_obj, *args, **kwargs)
    except Exception as e:
        if token:
            detach_span_from_context(token)
        if span:
            _end_span_err(mlclient, span, e)
        raise

    if token:
        detach_span_from_context(token)
    if span:
        _end_span_ok(mlclient, span, outputs=result)
    return result


def _patched_run_sync(original, *args, **kwargs):
    _get_or_create_session_span()

    cfg = AutoLoggingConfig.init(flavor_name=FLAVOUR_NAME)
    mlclient = mlflow.MlflowClient()
    active = mlflow.active_run()
    run_id = active.info.run_id if active else None
    self_obj, *call_args = args
    inputs = construct_full_inputs(original, self_obj, *call_args, **kwargs)

    span = None
    token = None
    if cfg.log_traces:
        span = _start_span(
            mlclient,
            name=f"{self_obj.__class__.__name__}.run_sync",
            span_type=SpanType.CHAIN,
            inputs=inputs,
            run_id=run_id,
        )
        token = set_span_in_context(span)
    try:
        output = original(self_obj, *call_args, **kwargs)
    except Exception as e:
        if token:
            detach_span_from_context(token)
        if span:
            _end_span_err(mlclient, span, e)
        raise
    if token:
        detach_span_from_context(token)
    if span:
        _end_span_ok(mlclient, span, outputs=output)
    return output


async def _patched_tool_run(original, self_obj, message, run_context, tracer):
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOUR_NAME)
    if not cfg.log_traces:
        return await original(self_obj, message, run_context, tracer)

    mlclient = mlflow.MlflowClient()
    tool_name = self_obj.name

    # TODO: remove this when we have a better way to get the inputs
    inputs = {
        "tool_name": self_obj.name,
        "tool_call_id": message.tool_call_id,
        "tool_arguments": json.loads(message.args_as_json_str()),
        "run_context": {
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
        },
    }
    span = start_client_span_or_trace(
        mlclient, name=f"{tool_name}", span_type=SpanType.TOOL, inputs=inputs
    )
    try:
        result = await original(self_obj, message, run_context, tracer)
    except Exception as e:
        span.add_event(SpanEvent.from_exception(e))
        mlclient.end_span(span.request_id, span.span_id, status=SpanStatusCode.ERROR)
        raise
    end_client_span_or_trace(mlclient, span, outputs=result)
    return result


async def _patched_llm_request(original: Any, self_obj: Any, *args: Any, **kwargs: Any) -> Any:
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOUR_NAME)
    if not (cfg and cfg.log_traces):
        return await original(self_obj, *args, **kwargs)

    mlclient = mlflow.MlflowClient()
    inputs = construct_full_inputs(original, self_obj, *args, **kwargs)
    if hasattr(self_obj, "agent"):
        inputs["_state"] = deepcopy(getattr(self_obj.agent, "_state", {}))

    backend = getattr(self_obj, "wrapped", self_obj)
    backend_name = backend.__class__.__name__
    span = _start_span(
        mlclient,
        name=f"{backend_name}.request",
        span_type=SpanType.LLM,
        inputs=inputs,
        run_id=None,
    )
    try:
        result = await original(self_obj, *args, **kwargs)
    except Exception as e:
        _end_span_err(mlclient, span, e)
        raise
    _end_span_ok(mlclient, span, outputs=result)
    return result


async def _patched_llm_request_stream(
    original: Any, self_obj: Any, *args: Any, **kwargs: Any
) -> AsyncIterator[Any]:
    cfg = AutoLoggingConfig.get()
    if not (cfg and cfg.log_traces):
        async for chunk in original(self_obj, *args, **kwargs):
            yield chunk
        return

    mlclient = mlflow.MlflowClient()
    inputs = construct_full_inputs(original, self_obj, *args, **kwargs)
    if hasattr(self_obj, "agent"):
        inputs["_state"] = deepcopy(getattr(self_obj.agent, "_state", {}))

    span = _start_span(
        mlclient,
        name=f"{self_obj.__class__.__name__}.request_stream",
        span_type=SpanType.LLM,
        inputs=inputs,
        run_id=None,
    )
    try:
        async for chunk in original(self_obj, *args, **kwargs):
            yield chunk
    except Exception as e:
        _end_span_err(mlclient, span, e)
        raise
    _end_span_ok(mlclient, span, outputs=None)


async def _patched_mcp_call_tool(original, self_obj, tool_name, arguments):
    _get_or_create_session_span()
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOUR_NAME)

    if not (cfg and cfg.log_traces):
        return await original(self_obj, tool_name, arguments)

    mlclient = mlflow.MlflowClient()
    span = _start_span(
        mlclient,
        name=f"{self_obj.__class__.__name__}.call_tool",
        span_type=SpanType.TOOL,
        inputs={"tool_name": tool_name, "arguments": arguments},
        run_id=mlflow.active_run().info.run_id,
    )
    try:
        result = await original(self_obj, tool_name, arguments)
    except Exception as e:
        _end_span_err(mlclient, span, e)
        raise
    _end_span_ok(mlclient, span, outputs=result)
    return result


async def _patched_mcp_list_tools(original, self_obj):
    _get_or_create_session_span()
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOUR_NAME)

    if not (cfg and cfg.log_traces):
        return await original(self_obj)

    mlclient = mlflow.MlflowClient()
    span = _start_span(
        mlclient,
        name=f"{self_obj.__class__.__name__}.list_tools",
        span_type=SpanType.CHAIN,
        inputs={},
        run_id=mlflow.active_run().info.run_id,
    )
    try:
        tools = await original(self_obj)
    except Exception as e:
        _end_span_err(mlclient, span, e)
        raise
    _end_span_ok(mlclient, span, outputs=[t.name for t in tools])
    return tools


@autologging_integration(FLAVOUR_NAME)
def autolog(
    *,
    log_traces: bool = True,
    extra_tags: Optional[dict[str, str]] = None,
    disable: bool = False,
    silent: bool = False,
) -> None:
    import pydantic_ai
    from pydantic_ai.mcp import MCPServer
    from pydantic_ai.models.instrumented import InstrumentedModel
    from pydantic_ai.tools import Tool

    if disable:
        return

    AutoLoggingConfig.init(flavor_name=FLAVOUR_NAME)

    safe_patch(FLAVOUR_NAME, _fluent, "end_run", _patched_end_run)
    safe_patch(FLAVOUR_NAME, mlflow, "end_run", _patched_end_run)

    safe_patch(FLAVOUR_NAME, pydantic_ai.Agent, "run_sync", _patched_run_sync)
    safe_patch(FLAVOUR_NAME, pydantic_ai.Agent, "run", _patched_run)

    safe_patch(FLAVOUR_NAME, InstrumentedModel, "request", _patched_llm_request)
    if hasattr(InstrumentedModel, "request_stream"):
        safe_patch(FLAVOUR_NAME, InstrumentedModel, "request_stream", _patched_llm_request_stream)
    safe_patch(FLAVOUR_NAME, Tool, "run", _patched_tool_run)

    safe_patch(FLAVOUR_NAME, MCPServer, "call_tool", _patched_mcp_call_tool)
    safe_patch(FLAVOUR_NAME, MCPServer, "list_tools", _patched_mcp_list_tools)
