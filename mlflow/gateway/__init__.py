"""
.. warning::

    MLflow AI gateway is deprecated and has been replaced by the deployments API for generative AI.
    See :ref:`gateway-migration` for migration.
"""
from mlflow.gateway.client import MlflowGatewayClient
from mlflow.gateway.fluent import (
    create_route,
    delete_route,
    get_limits,
    get_route,
    query,
    search_routes,
    set_limits,
)
from mlflow.gateway.utils import get_gateway_uri, set_gateway_uri

__all__ = [
    "create_route",
    "delete_route",
    "get_route",
    "set_limits",
    "get_limits",
    "get_gateway_uri",
    "MlflowGatewayClient",
    "query",
    "search_routes",
    "set_gateway_uri",
]
