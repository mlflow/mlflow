import functools
import inspect
from collections.abc import Callable
from typing import Any

import mlflow
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.gateway.schemas.chat import StreamResponsePayload
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig


def _maybe_unwrap_single_arg_input(args, kwargs):
    """Unwrap single-argument inputs so trace shows the request body directly"""
    if len(args) == 1 and not kwargs:
        if span := mlflow.get_current_active_span():
            span.set_inputs(args[0])


def maybe_traced_gateway_call(
    func: Callable[..., Any],
    endpoint_config: GatewayEndpointConfig,
    metadata: dict[str, Any] | None = None,
    output_reducer: Callable[[list[Any]], Any] | None = None,
) -> Callable[..., Any]:
    """
    Wrap a gateway function with tracing.

    Args:
        func: The function to trace.
        endpoint_config: The gateway endpoint configuration.
        metadata: Additional metadata to include in the trace (e.g., auth user info).
        output_reducer: A function to aggregate streaming chunks into a single output.

    Returns:
        A traced version of the function.

    Usage:
        result = await traced_gateway_call(provider.chat, endpoint_config)(payload)
    """
    if not endpoint_config.experiment_id:
        return func

    trace_kwargs = {
        "name": f"gateway/{endpoint_config.endpoint_name}",
        "attributes": {
            "endpoint_id": endpoint_config.endpoint_id,
            "endpoint_name": endpoint_config.endpoint_name,
        },
        "output_reducer": output_reducer,
        "trace_destination": MlflowExperimentLocation(endpoint_config.experiment_id),
    }

    if inspect.isasyncgenfunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            _maybe_unwrap_single_arg_input(args, kwargs)
            if metadata:
                mlflow.update_current_trace(metadata=metadata)
            async for item in func(*args, **kwargs):
                yield item

    elif inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            _maybe_unwrap_single_arg_input(args, kwargs)
            if metadata:
                mlflow.update_current_trace(metadata=metadata)
            return await func(*args, **kwargs)

    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _maybe_unwrap_single_arg_input(args, kwargs)
            if metadata:
                mlflow.update_current_trace(metadata=metadata)
            return func(*args, **kwargs)

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
