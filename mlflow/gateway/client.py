import logging
import requests
from typing import Optional
from urllib.parse import urljoin

from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_BASE
from mlflow.gateway.utils import (
    _is_valid_uri,
    _is_gateway_server_available,
    validate_gateway_uri_is_set,
    _get_gateway_response_with_retries,
)

_logger = logging.getLogger(__name__)


def set_mlflow_gateway_uri(gateway_uri: str):
    if not _is_valid_uri(gateway_uri):
        raise MlflowException.invalid_parameter_value(
            "The gateway uri provided is missing required elements. Ensure that the schema "
            "and netloc are provided."
        )

    if not _is_gateway_server_available(gateway_uri):
        raise MlflowException.invalid_parameter_value(
            f"The gateway server cannot be verified at {gateway_uri}. Please verify that the "
            "server has been started and that you are able to ping it."
        )

    MLFLOW_GATEWAY_URI.set(gateway_uri)


@validate_gateway_uri_is_set
def get_mlflow_gateway_uri():
    return MLFLOW_GATEWAY_URI.get()


@validate_gateway_uri_is_set
def get_route(name: str):
    base_route_url = urljoin(MLFLOW_GATEWAY_URI.get(), MLFLOW_GATEWAY_ROUTE_BASE)
    route_url = urljoin(base_route_url, name)
    return _get_gateway_response_with_retries(method="GET", url=route_url).json()


@validate_gateway_uri_is_set
def search_routes(filter: Optional[str] = None):
    if filter:
        _logger.warning(
            "Search functionality is not implemented. This API will list " "all configured routes."
        )
    search_url = urljoin(MLFLOW_GATEWAY_URI.get(), MLFLOW_GATEWAY_ROUTE_BASE)
    return _get_gateway_response_with_retries(method="GET", url=search_url).json()
