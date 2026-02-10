import functools
import inspect
from collections.abc import Callable
from typing import Any

import mlflow
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig


def maybe_traced_gateway_call(
    func: Callable[..., Any],
    endpoint_config: GatewayEndpointConfig,
    metadata: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """
    Wrap a gateway function with tracing.

    Args:
        func: The function to trace.
        endpoint_config: The gateway endpoint configuration.
        metadata: Additional metadata to include in the trace (e.g., auth user info).

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
        "trace_destination": MlflowExperimentLocation(endpoint_config.experiment_id),
    }

    if not metadata:
        return mlflow.trace(func, **trace_kwargs)

    # Wrap function to set metadata inside the trace context
    if inspect.isasyncgenfunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            mlflow.update_current_trace(metadata=metadata)
            async for item in func(*args, **kwargs):
                yield item

    elif inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            mlflow.update_current_trace(metadata=metadata)
            return await func(*args, **kwargs)

    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mlflow.update_current_trace(metadata=metadata)
            return func(*args, **kwargs)

    return mlflow.trace(wrapper, **trace_kwargs)
