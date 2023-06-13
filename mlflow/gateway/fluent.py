import logging
from typing import Optional
from urllib.parse import urljoin

from mlflow.exceptions import MlflowException
from mlflow.gateway.envs import MLFLOW_GATEWAY_URI  # TODO: change to environment_variables import
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_BASE
from mlflow.gateway.utils import _validate_gateway_uri_is_set

_logger = logging.getLogger(__name__)


# TODO: THIS IS A WIP. WILL CHANGE IN NEXT COMMIT!


# @_validate_gateway_uri_is_set
# def get_route(name: str):
#     # TODO: use the MlflowGatewayClient
#
#     base_route_url = urljoin(MLFLOW_GATEWAY_URI.get(), MLFLOW_GATEWAY_ROUTE_BASE)
#     route_url = urljoin(base_route_url, name)
#     return _get_gateway_response_with_retries(method="GET", url=route_url).json()
#
#
# @_validate_gateway_uri_is_set
# def search_routes(search_filter: Optional[str] = None):
#     # TODO: use the MlflowGatewayClient
#
#     if search_filter is not None:
#         raise MlflowException.invalid_parameter_value(
#             "Search functionality is not implemented. This API only returns all configured routes "
#             "with no `search_filter` defined."
#         )
#     search_url = urljoin(MLFLOW_GATEWAY_URI.get(), MLFLOW_GATEWAY_ROUTE_BASE)
#     return _get_gateway_response_with_retries(method="GET", url=search_url).json()
