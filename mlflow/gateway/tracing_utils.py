import dataclasses
import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

import mlflow
from mlflow.entities import SpanStatus, SpanType
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.gateway.config import GatewayRequestType
from mlflow.gateway.schemas.chat import StreamResponsePayload
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey
from mlflow.tracing.distributed import set_tracing_context_from_http_request_headers
from mlflow.tracing.trace_manager import InMemoryTraceManager

_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _ModelSpanInfo:
    name: str
    attributes: dict[str, Any]
    status: SpanStatus | None = None
    start_time_ns: int | None = None
    end_time_ns: int | None = None


def _maybe_unwrap_single_arg_input(args: tuple[Any], kwargs: dict[str, Any]):
    """Unwrap inputs so trace shows the request body directly.

    Extracts the payload kwarg if present, otherwise unwraps single-argument inputs.
    """
    span = mlflow.get_current_active_span()
    if not span:
        return

    # For passthrough endpoints with kwargs, extract the payload key
    # This takes precedence to handle cases where functions are called with
    # keyword arguments (e.g., action=..., payload=..., headers=...)
    if "payload" in kwargs:
        span.set_inputs(kwargs["payload"])
    # For other endpoints with a single positional argument
    elif len(args) == 1 and not kwargs:
        span.set_inputs(args[0])


def _has_traceparent(headers: dict[str, str]) -> bool:
    return "traceparent" in headers or "Traceparent" in headers


def _gateway_span_name(endpoint_config: GatewayEndpointConfig) -> str:
    return f"gateway/{endpoint_config.endpoint_name}"


def _gateway_span_attributes(
    endpoint_config: GatewayEndpointConfig,
    request_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    attrs = {
        "endpoint_id": endpoint_config.endpoint_id,
        "endpoint_name": endpoint_config.endpoint_name,
    }
    if request_headers:
        if host := request_headers.get("host") or request_headers.get("Host"):
            attrs["server_url"] = host
    return attrs


_MODEL_SPAN_ATTRIBUTE_KEYS = [
    SpanAttributeKey.CHAT_USAGE,
    SpanAttributeKey.MODEL,
    SpanAttributeKey.MODEL_PROVIDER,
]


def _get_model_span_info(gateway_trace_id: str) -> list[_ModelSpanInfo]:
    """Read name and attributes from non-root model spans within a gateway trace."""
    trace_manager = InMemoryTraceManager.get_instance()
    results: list[_ModelSpanInfo] = []
    with trace_manager.get_trace(gateway_trace_id) as trace:
        if trace is None:
            return results
        for span in trace.span_dict.values():
            if span.parent_id is None:
                continue
            attrs = {}
            for key in _MODEL_SPAN_ATTRIBUTE_KEYS:
                if value := span.get_attribute(key):
                    attrs[key] = value
            if attrs:
                results.append(
                    _ModelSpanInfo(
                        name=span.name,
                        attributes=attrs,
                        status=span.status,
                        start_time_ns=span.start_time_ns,
                        end_time_ns=span.end_time_ns,
                    )
                )
    return results


def _maybe_create_distributed_span(
    request_headers: dict[str, str] | None,
    endpoint_config: GatewayEndpointConfig,
) -> None:
    """Create lightweight mirror spans under the caller's distributed trace.

    When a ``traceparent`` header is present the gateway already records a
    full trace (with payloads) in its own experiment.  This helper attaches a
    *summary* to the caller's trace so that the caller can see gateway
    activity without duplicating large request/response bodies.

    The resulting shape in the caller's trace looks like::

        [caller span]
          └── gateway/<endpoint>          # attributes: endpoint info + linked trace id
                ├── model/<provider>/<m>   # attributes: usage, model, status
                └── model/<provider>/<m>   # (one per non-root gateway span)
    """
    if not request_headers or not _has_traceparent(request_headers):
        return

    gateway_trace_id = None
    if span := mlflow.get_current_active_span():
        gateway_trace_id = span.trace_id

    model_infos = _get_model_span_info(gateway_trace_id) if gateway_trace_id else []

    try:
        with set_tracing_context_from_http_request_headers(request_headers):
            with mlflow.start_span(
                name=_gateway_span_name(endpoint_config),
                span_type=SpanType.LLM,
            ) as gw_span:
                attrs = _gateway_span_attributes(endpoint_config, request_headers)
                if gateway_trace_id:
                    attrs[SpanAttributeKey.LINKED_GATEWAY_TRACE_ID] = gateway_trace_id
                gw_span.set_attributes(attrs)

                for info in model_infos:
                    model_span = mlflow.start_span_no_context(
                        name=info.name,
                        span_type=SpanType.LLM,
                        parent_span=gw_span,
                        attributes=info.attributes,
                        start_time_ns=info.start_time_ns,
                    )
                    model_span.end(
                        status=info.status,
                        end_time_ns=info.end_time_ns,
                    )
    except Exception:
        _logger.debug("Failed to create distributed trace span for gateway call", exc_info=True)


def maybe_traced_gateway_call(
    func: Callable[..., Any],
    endpoint_config: GatewayEndpointConfig,
    metadata: dict[str, Any] | None = None,
    output_reducer: Callable[[list[Any]], Any] | None = None,
    request_headers: dict[str, str] | None = None,
    request_type: GatewayRequestType | None = None,
) -> Callable[..., Any]:
    """
    Wrap a gateway function with tracing.

    Args:
        func: The function to trace.
        endpoint_config: The gateway endpoint configuration.
        metadata: Additional metadata to include in the trace (e.g., auth user info).
        output_reducer: A function to aggregate streaming chunks into a single output.
        request_headers: HTTP request headers; if they contain a traceparent header,
            a span will also be created under the agent's distributed trace.
        request_type: The type of gateway request (e.g., GatewayRequestType.CHAT).

    Returns:
        A traced version of the function.

    Usage:
        result = await traced_gateway_call(provider.chat, endpoint_config)(payload)
    """
    if not endpoint_config.experiment_id:
        return func

    trace_kwargs = {
        "name": _gateway_span_name(endpoint_config),
        "attributes": _gateway_span_attributes(endpoint_config, request_headers),
        "output_reducer": output_reducer,
        "trace_destination": MlflowExperimentLocation(endpoint_config.experiment_id),
    }

    # Build combined metadata with gateway-specific fields
    combined_metadata = metadata.copy() if metadata else {}
    combined_metadata[TraceMetadataKey.GATEWAY_ENDPOINT_ID] = endpoint_config.endpoint_id
    if request_type:
        combined_metadata[TraceMetadataKey.GATEWAY_REQUEST_TYPE] = request_type

    # Wrap function to set metadata inside the trace context
    if inspect.isasyncgenfunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            mlflow.update_current_trace(metadata=combined_metadata)
            _maybe_unwrap_single_arg_input(args, kwargs)
            try:
                async for item in func(*args, **kwargs):
                    yield item
            finally:
                _maybe_create_distributed_span(request_headers, endpoint_config)

    elif inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            mlflow.update_current_trace(metadata=combined_metadata)
            _maybe_unwrap_single_arg_input(args, kwargs)
            try:
                result = await func(*args, **kwargs)
            finally:
                _maybe_create_distributed_span(request_headers, endpoint_config)
            return result

    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mlflow.update_current_trace(metadata=combined_metadata)
            _maybe_unwrap_single_arg_input(args, kwargs)
            try:
                result = func(*args, **kwargs)
            finally:
                _maybe_create_distributed_span(request_headers, endpoint_config)
            return result

    return mlflow.trace(wrapper, **trace_kwargs)


def aggregate_chat_stream_chunks(chunks: list[StreamResponsePayload]) -> dict[str, Any] | None:
    """
    Aggregate streaming chat completion chunks into a single ChatCompletion-like response.

    Returns:
        A ChatCompletion-like response.
    """
    if not chunks:
        return None

    # Group state per choice index
    choices_state: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        for choice in chunk.choices:
            state = choices_state.setdefault(
                choice.index,
                {
                    "role": None,
                    "content_parts": [],
                    "tool_calls_by_index": {},
                    "finish_reason": None,
                },
            )
            delta = choice.delta
            if delta.role and state["role"] is None:
                state["role"] = delta.role
            if delta.content:
                state["content_parts"].append(delta.content)
            if choice.finish_reason:
                state["finish_reason"] = choice.finish_reason
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    tc = state["tool_calls_by_index"].setdefault(
                        tc_delta.index,
                        {"id": None, "type": "function", "name": "", "arguments": ""},
                    )
                    if tc_delta.id:
                        tc["id"] = tc_delta.id
                    if tc_delta.type:
                        tc["type"] = tc_delta.type
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["arguments"] += tc_delta.function.arguments

    aggregated_choices = []
    for choice_index, state in sorted(choices_state.items()):
        message: dict[str, Any] = {
            "role": state["role"] or "assistant",
            "content": "".join(state["content_parts"]) or None,
        }
        if state["tool_calls_by_index"]:
            message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in state["tool_calls_by_index"].values()
            ]
        aggregated_choices.append(
            {
                "index": choice_index,
                "message": message,
                "finish_reason": state["finish_reason"] or "stop",
            }
        )

    last_chunk = chunks[-1]
    result = {
        "id": last_chunk.id,
        "object": "chat.completion",
        "created": last_chunk.created,
        "model": last_chunk.model,
        "choices": aggregated_choices,
    }

    if last_chunk.usage:
        result["usage"] = {
            "prompt_tokens": last_chunk.usage.prompt_tokens,
            "completion_tokens": last_chunk.usage.completion_tokens,
            "total_tokens": last_chunk.usage.total_tokens,
        }

    return result
