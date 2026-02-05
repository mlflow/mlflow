from collections.abc import Callable
from typing import Any

import mlflow
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig


def maybe_traced_gateway_call(
    func: Callable[..., Any],
    endpoint_config: GatewayEndpointConfig,
) -> Callable[..., Any]:
    """
    Wrap a gateway function with tracing.

    Args:
        func: The function to trace.
        endpoint_config: The gateway endpoint configuration.

    Returns:
        A traced version of the function.

    Usage:
        result = await traced_gateway_call(provider.chat, endpoint_config)(payload)
    """
    if not endpoint_config.experiment_id:
        return func

    return mlflow.trace(
        func,
        name=f"gateway/{endpoint_config.endpoint_name}",
        attributes={
            "endpoint_id": endpoint_config.endpoint_id,
            "endpoint_name": endpoint_config.endpoint_name,
        },
        trace_destination=MlflowExperimentLocation(experiment_id=endpoint_config.experiment_id),
    )
