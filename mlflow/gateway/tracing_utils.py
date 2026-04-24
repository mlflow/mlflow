import dataclasses
import functools
import inspect
import json
import logging
from collections.abc import Callable
from typing import Any

import mlflow
from mlflow.entities import SpanStatus, SpanType
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.gateway.config import GatewayRequestType
from mlflow.gateway.schemas.chat import StreamResponsePayload
from mlflow.gateway.utils import parse_sse_lines
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
    SpanAttributeKey.LLM_COST,
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
    on_complete: Callable[[], None] | None = None,
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
        on_complete: A no-arg callback invoked inside the trace context after the
            provider call completes (in ``finally``).

    Returns:
        A traced version of the function.

    Usage:
        result = await traced_gateway_call(provider.chat, endpoint_config)(payload)
    """
    if not endpoint_config.usage_tracking:
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
                if on_complete:
                    try:
                        on_complete()
                    except Exception:
                        _logger.debug("on_complete callback failed", exc_info=True)
                _maybe_create_distributed_span(request_headers, endpoint_config)

    elif inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            mlflow.update_current_trace(metadata=combined_metadata)
            _maybe_unwrap_single_arg_input(args, kwargs)
            try:
                result = await func(*args, **kwargs)
            finally:
                if on_complete:
                    try:
                        on_complete()
                    except Exception:
                        _logger.debug("on_complete callback failed", exc_info=True)
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
                if on_complete:
                    try:
                        on_complete()
                    except Exception:
                        _logger.debug("on_complete callback failed", exc_info=True)
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
        aggregated_choices.append({
            "index": choice_index,
            "message": message,
            "finish_reason": state["finish_reason"] or "stop",
        })

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


def aggregate_anthropic_messages_stream_chunks(
    chunks: list[bytes],
) -> dict[str, Any] | None:
    """
    Aggregate raw Anthropic Messages API SSE streaming chunks into a single Messages response.

    Processes the following Anthropic streaming event types:
    - ``message_start``: extracts id, model, role, and input token usage
    - ``content_block_start``: initialises text or tool_use content blocks
    - ``content_block_delta``: appends text deltas and tool input JSON deltas
    - ``message_delta``: extracts stop_reason, stop_sequence, and output token usage

    Returns a dict matching the Anthropic Messages API non-streaming response shape::

        {
            "id": "msg_...",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "..."},
                {"type": "tool_use", "id": "...", "name": "...", "input": {...}},
            ],
            "model": "...",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": N, "cache_read_input_tokens": C, "output_tokens": M},
        }

    Returns ``None`` if *chunks* is empty or contains no parseable events.
    """
    if not chunks:
        return None

    # Concatenate all raw bytes before parsing. The aiohttp streaming iterator
    # yields arbitrary-sized byte chunks that can split a single SSE "data:" line
    # across multiple pieces; parse_sse_lines() requires complete lines. Joining
    # here ensures no events are silently dropped due to mid-line splits.
    combined = b"".join(chunks)

    msg_id: str | None = None
    model: str | None = None
    role: str = "assistant"
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: dict[str, Any] = {}
    # Ordered dict keyed by content block index preserving insertion order
    content_blocks: dict[int, dict[str, Any]] = {}

    for event in parse_sse_lines(combined):
        match event:
            case {"type": "message_start", "message": dict(msg)}:
                msg_id = msg.get("id")
                model = msg.get("model")
                role = msg.get("role", "assistant")
                # Merge all usage fields (input_tokens, cache_read_input_tokens,
                # cache_creation_input_tokens, …) present in message_start.
                if msg_usage := msg.get("usage"):
                    usage.update({k: v for k, v in msg_usage.items() if v is not None})
            case {
                "type": "content_block_start",
                "index": int(index),
                "content_block": dict(block),
            }:
                block_type = block.get("type")
                if block_type == "tool_use":
                    content_blocks[index] = {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "_input_json": "",
                    }
                else:
                    content_blocks[index] = {"type": "text", "text": block.get("text", "")}
            case {
                "type": "content_block_delta",
                "index": int(index),
                "delta": dict(delta),
            }:
                block = content_blocks.get(index)
                if block is None:
                    continue
                match delta.get("type"):
                    case "text_delta":
                        block["text"] = block.get("text", "") + delta.get("text", "")
                    case "input_json_delta":
                        block["_input_json"] = block.get("_input_json", "") + delta.get(
                            "partial_json", ""
                        )
            case {"type": "message_delta", "delta": dict(delta)}:
                stop_reason = delta.get("stop_reason", stop_reason)
                stop_sequence = delta.get("stop_sequence", stop_sequence)
                # Merge output_tokens (and any extra fields) from message_delta.
                if delta_usage := event.get("usage"):
                    usage.update({k: v for k, v in delta_usage.items() if v is not None})

    if msg_id is None and not content_blocks:
        return None

    # Finalise content blocks: parse accumulated tool input JSON
    content: list[dict[str, Any]] = []
    for block in (content_blocks[i] for i in sorted(content_blocks)):
        if block["type"] == "tool_use":
            raw_json = block.pop("_input_json", "")
            try:
                block["input"] = json.loads(raw_json) if raw_json else {}
            except json.JSONDecodeError:
                block["input"] = {}
        content.append(block)

    result: dict[str, Any] = {
        "id": msg_id,
        "type": "message",
        "role": role,
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
    }
    if usage:
        result["usage"] = usage

    return result


def aggregate_gemini_stream_generate_content_chunks(
    chunks: list[bytes],
) -> dict[str, Any] | None:
    """
    Aggregate raw Gemini ``streamGenerateContent`` SSE chunks into a single response.

    Each streaming event is a complete JSON object in the Gemini
    ``GenerateContentResponse`` format. Text parts are concatenated across all events;
    function-call parts and metadata (``finishReason``, ``usageMetadata``) are taken
    from the last event that carries them.

    Returns a dict matching the Gemini non-streaming ``generateContent`` response shape::

        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "..."},
                            {"functionCall": {"name": "...", "args": {...}}},
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": N,
                "candidatesTokenCount": M,
                "totalTokenCount": T,
            },
        }

    Returns ``None`` if *chunks* is empty or contains no parseable events.
    """
    if not chunks:
        return None

    # Concatenate before parsing: aiohttp yields arbitrary-sized byte chunks that
    # can split a single SSE "data:" line across multiple pieces.
    combined = b"".join(chunks)

    # candidate index → accumulated state
    candidates_state: dict[int, dict[str, Any]] = {}
    usage_metadata: dict[str, Any] | None = None

    for event in parse_sse_lines(combined):
        for cand_idx, candidate in enumerate(event.get("candidates", [])):
            idx = candidate.get("index", cand_idx)
            state = candidates_state.setdefault(
                idx,
                {
                    "role": "model",
                    "text_parts": [],
                    "function_call_parts": [],
                    "finish_reason": None,
                },
            )
            content = candidate.get("content", {})
            if role := content.get("role"):
                state["role"] = role
            for part in content.get("parts", []):
                if "text" in part:
                    state["text_parts"].append(part["text"])
                elif "functionCall" in part:
                    state["function_call_parts"].append(part["functionCall"])
            if finish_reason := candidate.get("finishReason"):
                state["finish_reason"] = finish_reason
        if um := event.get("usageMetadata"):
            usage_metadata = um

    if not candidates_state:
        return None

    candidates = []
    for idx, state in sorted(candidates_state.items()):
        parts: list[dict[str, Any]] = []
        if text := "".join(state["text_parts"]):
            parts.append({"text": text})
        parts.extend({"functionCall": fc} for fc in state["function_call_parts"])
        candidates.append({
            "content": {"parts": parts, "role": state["role"]},
            "finishReason": state["finish_reason"],
            "index": idx,
        })

    result: dict[str, Any] = {"candidates": candidates}
    if usage_metadata:
        result["usageMetadata"] = usage_metadata
    return result


def aggregate_openai_responses_stream_chunks(
    chunks: list[bytes],
) -> dict[str, Any] | None:
    """
    Aggregate raw OpenAI Responses API SSE streaming chunks into a single response object.

    The OpenAI Responses streaming API emits a ``response.completed`` event that contains
    the fully-assembled response object — including all output items, content parts, and
    token usage. This function locates that event and returns its ``response`` field,
    giving the same shape as a non-streaming Responses API call::

        {
            "id": "resp_...",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "..."}],
                }
            ],
            "usage": {"input_tokens": N, "output_tokens": M, "total_tokens": T},
            ...
        }

    Returns ``None`` if *chunks* is empty or contains no ``response.completed`` event.
    """
    if not chunks:
        return None

    # Scan chunks incrementally to avoid materializing a second full copy of the
    # stream bytes.  aiohttp yields arbitrary-sized byte chunks that can bisect a
    # ``data:`` line, so we carry any trailing incomplete line into the next
    # iteration rather than joining everything up front.
    leftover = b""
    for chunk in chunks:
        data = leftover + chunk
        # Split on newlines, keeping the last (potentially incomplete) segment.
        lines = data.split(b"\n")
        leftover = lines[-1]
        complete = b"\n".join(lines[:-1]) + b"\n"
        for event in parse_sse_lines(complete):
            if event.get("type") == "response.completed":
                return event.get("response")

    # Flush any remaining bytes that were not followed by a newline.
    if leftover:
        for event in parse_sse_lines(leftover):
            if event.get("type") == "response.completed":
                return event.get("response")

    return None
