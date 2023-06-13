import json
import logging
from urllib.parse import urljoin
from typing import Optional

from mlflow.gateway.config import Route
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_BASE, MLFLOW_GATEWAY_HEALTH_ENDPOINT
from mlflow.gateway.utils import (
    _resolve_gateway_uri,
    _get_gateway_response_with_retries,
    _merge_uri_paths,
)


_logger = logging.getLogger(__name__)


class MlflowGatewayClient:
    """
    Client for interacting with the MLflow Gateway API.

    :param gateway_uri: Optional URI of the gateway. If not provided, attempts to resolve from
    first the stored result of `set_gateway_uri()`, then the  environment variable
    `MLFLOW_GATEWAY_URI`.
    """

    def __init__(self, gateway_uri: Optional[str] = None):
        self._gateway_uri = _resolve_gateway_uri(gateway_uri)
        self._route_base = MLFLOW_GATEWAY_ROUTE_BASE

    @property
    def get_gateway_uri(self):
        """
        Get the current value for the URI of the MLflow Gateway.

        :return: The gateway URI.
        """
        return self._gateway_uri

    def _call_endpoint(self, method: str, route: str, json_body: Optional[str] = None):
        """
        Call a specific endpoint on the Gateway API.

        :param method: The HTTP method to use.
        :param route: The API route to call.
        :param json_body: Optional JSON body to include in the request.
        :return: The server's response.
        """
        if json_body:
            json_body = json.loads(json_body)

        url = urljoin(self._gateway_uri, route)

        call_kwargs = {
            "method": method,
            "url": url,
        }
        if method.lower() == "get":
            call_kwargs["params"] = json_body
        else:
            call_kwargs["json"] = json_body

        return _get_gateway_response_with_retries(**call_kwargs)

    def get_gateway_health(self):
        """
        Get the health status of the Gateway. This returns the state of the underlying FastAPI
        app that is running on the uvicorn server, managed by the gunicorn process manager.

        A standard health response will return {"status":"OK"} if the server is up and ready to
        process requests sent to it.

        :return: The JSON response from the server.
        """
        url = urljoin(self._gateway_uri, MLFLOW_GATEWAY_HEALTH_ENDPOINT)
        return self._call_endpoint("GET", url).json()

    def get_route(self, name: str):
        """
        Get a specific query route from the gateway. The routes that are available to retrieve
        are only those that have been configured through the MLflow Gateway Server configuration
        file (set during server start or through server update commands).

        :param name: The name of the route.
        :return: The returned data structure is a serialized representation of the `Route` data
        structure, giving information about the name, type, and model details (model name
        and provider) for the requested route endpoint.
        """
        url = urljoin(self._gateway_uri, _merge_uri_paths([self._route_base, name]))
        response = self._call_endpoint("GET", url).json()
        return Route(**response["route"])

    def search_routes(self, search_filter: Optional[str] = None):
        """
        Search for routes in the Gateway. Currently, this simply returns all configured routes.

        :param search_filter: An optional filter to apply to the search. Currently not used.
        :return: Returns a list of all configured and initialized `Route` data for the MLflow
        Gateway Server. The return will be a list of dictionaries that detail the name, type,
        and model details of each active route endpoint.
        """
        if search_filter:
            _logger.warning(
                "Search functionality is not implemented. This API will list all configured routes."
            )
        url = urljoin(self._gateway_uri, self._route_base)
        response = self._call_endpoint("GET", url).json()["routes"]
        return [Route(**resp) for resp in response]
