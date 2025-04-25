# _pydanticai_autolog.py

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, AsyncIterator, Optional

import pydantic_ai
from pydantic_ai.models.instrumented import InstrumentedModel

import mlflow
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
FLAVOUR_NAME: str = "pydanticai"


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


def _patched_run_sync(original, *args, **kwargs):
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


@autologging_integration(FLAVOUR_NAME)
def autolog(
    *,
    log_traces: bool = True,
    extra_tags: Optional[dict[str, str]] = None,
    disable: bool = False,
    silent: bool = False,
) -> None:
    if disable:
        return

    AutoLoggingConfig.init(flavor_name=FLAVOUR_NAME)

    safe_patch(FLAVOUR_NAME, pydantic_ai.Agent, "run_sync", _patched_run_sync)

    safe_patch(FLAVOUR_NAME, InstrumentedModel, "request", _patched_llm_request)
    if hasattr(InstrumentedModel, "request_stream"):
        safe_patch(FLAVOUR_NAME, InstrumentedModel, "request_stream", _patched_llm_request_stream)
