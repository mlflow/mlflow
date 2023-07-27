import json
import logging
from typing import Optional, Dict, Any

import requests.exceptions

from mlflow import MlflowException
from mlflow.gateway.config import Route
from mlflow.gateway.constants import (
    MLFLOW_GATEWAY_CRUD_ROUTE_BASE,
    MLFLOW_GATEWAY_ROUTE_BASE,
    MLFLOW_QUERY_SUFFIX,
    MLFLOW_GATEWAY_CLIENT_QUERY_TIMEOUT_SECONDS,
    MLFLOW_GATEWAY_CLIENT_QUERY_RETRY_CODES,
)
from mlflow.gateway.utils import get_gateway_uri, assemble_uri_path, resolve_route_url
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracking._tracking_service.utils import _get_default_host_creds
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import MlflowHostCreds, http_request, augmented_raise_for_status
from mlflow.utils.uri import get_uri_scheme


_logger = logging.getLogger(__name__)


@experimental
class MlflowGatewayClient:
    """
    Client for interacting with the MLflow Gateway API.

    :param gateway_uri: Optional URI of the gateway. If not provided, attempts to resolve from
        first the stored result of `set_gateway_uri()`, then the  environment variable
        `MLFLOW_GATEWAY_URI`.
    """

    def __init__(self, gateway_uri: Optional[str] = None):
        self._gateway_uri = gateway_uri or get_gateway_uri()
        self._host_creds = self._resolve_host_creds()

    def _is_databricks_host(self) -> bool:
        return (
            self._gateway_uri == "databricks" or get_uri_scheme(self._gateway_uri) == "databricks"
        )

    def _resolve_host_creds(self) -> MlflowHostCreds:
        if self._is_databricks_host():
            return get_databricks_host_creds(self._gateway_uri)
        else:
            return _get_default_host_creds(self._gateway_uri)

    @property
    def gateway_uri(self):
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

        call_kwargs = {}
        if method.lower() == "get":
            call_kwargs["params"] = json_body
        else:
            call_kwargs["json"] = json_body

        response = http_request(
            host_creds=self._host_creds,
            endpoint=route,
            method=method,
            timeout=MLFLOW_GATEWAY_CLIENT_QUERY_TIMEOUT_SECONDS,
            retry_codes=MLFLOW_GATEWAY_CLIENT_QUERY_RETRY_CODES,
            **call_kwargs,
        )
        augmented_raise_for_status(response)
        return response

    @experimental
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
        route = assemble_uri_path([MLFLOW_GATEWAY_CRUD_ROUTE_BASE, name])
        response = self._call_endpoint("GET", route).json()
        response["route_url"] = resolve_route_url(self._gateway_uri, response["route_url"])

        return Route(**response)

    @experimental
    def search_routes(self, page_token: Optional[str] = None) -> PagedList[Route]:
        """
        Search for routes in the Gateway.

        :param page_token: Token specifying the next page of results. It should be obtained from
                           a prior ``search_routes()`` call.
        :return: Returns a list of all configured and initialized `Route` data for the MLflow
            Gateway Server. The return will be a list of dictionaries that detail the name, type,
            and model details of each active route endpoint.
        """
        request_parameters = {"page_token": page_token} if page_token is not None else None
        response_json = self._call_endpoint(
            "GET", MLFLOW_GATEWAY_CRUD_ROUTE_BASE, json_body=json.dumps(request_parameters)
        ).json()
        for route in response_json["routes"]:
            route["route_url"] = resolve_route_url(self._gateway_uri, route["route_url"])

        routes = [Route(**resp) for resp in response_json["routes"]]
        next_page_token = response_json.get("next_page_token")
        return PagedList(routes, next_page_token)

    @experimental
    def create_route(self, name: str, route_type: str, model: Dict[str, Any]) -> Route:
        """
        Create a new route in the Gateway.

        .. warning::

            This API is **only available** when running within Databricks. When running elsewhere,
            route configuration is handled via updates to the route configuration YAML file that
            is specified during Gateway server start.

        :param name: The name of the route.
        :param route_type: The type of the route (e.g., 'llm/v1/chat', 'llm/v1/completions',
                           'llm/v1/embeddings').
        :param model: A dictionary representing the model details to be associated with the route.
                      This dictionary should define:

                      - The model name (e.g., "gpt-3.5-turbo")
                      - The provider (e.g., "openai", "anthropic")
                      - The configuration for the model used in the route

        :return: A serialized representation of the `Route` data structure,
                 providing information about the name, type, and model details for the
                 newly created route endpoint.

        :raises mlflow.MlflowException: If the function is not running within Databricks.

        .. note::

            See the official Databricks documentation for MLflow Gateway for examples of supported
            model configurations and how to dynamically create new routes within Databricks.


        Example usage from within Databricks:

        .. code-block:: python

            from mlflow.gateway import MlflowGatewayClient

            gateway_client = MlflowGatewayClient("databricks")

            openai_api_key = ...

            new_route = gateway_client.create_route(
                "my-new-route",
                "llm/v1/completions",
                {
                    "name": "question-answering-bot-1",
                    "provider": "openai",
                    "config": {
                        "openai_api_key": openai_api_key,
                        "openai_api_version": "2023-05-10",
                        "openai_api_type": "openai/v1/chat/completions",
                    },
                },
            )

        """
        if not self._is_databricks_host():
            raise MlflowException(
                "The create_route API is only available when running within "
                "Databricks. Route creation is handled through creating a "
                "configuration YAML file during startup or through updating a "
                "running Gateway server.",
                error_code=BAD_REQUEST,
            )
        payload = {
            "name": name,
            "route_type": route_type,
            "model": model,
        }
        response = self._call_endpoint(
            "POST", MLFLOW_GATEWAY_CRUD_ROUTE_BASE, json.dumps(payload)
        ).json()
        return Route(**response)

    @experimental
    def delete_route(self, name: str) -> None:
        """
        Delete an existing route in the Gateway.

        .. warning::

            This API is **only available** when running within Databricks. When running elsewhere,
            route deletion is handled by removing the corresponding entry from the route
            configuration YAML file that is specified during Gateway server start.

        :param name: The name of the route to delete.

        :raises mlflow.MlflowException: If the function is not running within Databricks.

        Example usage from within Databricks:

        .. code-block:: python

            from mlflow.gateway import MlflowGatewayClient

            gateway_client = MlflowGatewayClient("databricks")
            gateway_client.delete_route("my-existing-route")

        """
        if not self._is_databricks_host():
            raise MlflowException(
                "The delete_route API is only available when running within Databricks. Route "
                "deletion is handled through uploading a modified configuration YAML file to the "
                "location specified when starting the Gateway server. To delete a route, remove "
                "the route entry from the configuration file.",
                error_code=BAD_REQUEST,
            )
        route = assemble_uri_path([MLFLOW_GATEWAY_CRUD_ROUTE_BASE, name])
        self._call_endpoint("DELETE", route)

    @experimental
    def query(self, route: str, data: Dict[str, Any]):
        """
        Submit a query to a configured provider route.

        :param route: The name of the route to submit the query to.
        :param data: The data to send in the query. A dictionary representing the per-route
            specific structure required for a given provider.
        :return: The route's response as a dictionary, standardized to the route type.

        For chat, the structure should be:

        .. code-block:: python

            from mlflow.gateway import MlflowGatewayClient

            gateway_client = MlflowGatewayClient("http://my.gateway:8888")

            response = gateway_client.query(
                "my-chat-route",
                {"messages": [{"role": "user", "content": "Tell me a joke about rabbits"}, ...]},
            )

        For completions, the structure should be:

        .. code-block:: python

            from mlflow.gateway import MlflowGatewayClient

            gateway_client = MlflowGatewayClient("http://my.gateway:8888")

            response = gateway_client.query(
                "my-completions-route", {"prompt": "It's one small step for"}
            )

        For embeddings, the structure should be:

        .. code-block:: python

            from mlflow.gateway import MlflowGatewayClient

            gateway_client = MlflowGatewayClient("http://my.gateway:8888")

            response = gateway_client.query(
                "my-embeddings-route",
                {"text": ["It was the best of times", "It was the worst of times"]},
            )

        """

        data = json.dumps(data)

        query_route = assemble_uri_path([MLFLOW_GATEWAY_ROUTE_BASE, route, MLFLOW_QUERY_SUFFIX])

        try:
            return self._call_endpoint("POST", query_route, data).json()
        except MlflowException as e:
            if isinstance(e.__cause__, requests.exceptions.Timeout):
                timeout_message = (
                    "The provider has timed out while generating a response to your "
                    "query. Please evaluate the available parameters for the query "
                    "that you are submitting. Some parameter values and inputs can "
                    "increase the computation time beyond the allowable route "
                    f"timeout of {MLFLOW_GATEWAY_CLIENT_QUERY_TIMEOUT_SECONDS} "
                    "seconds."
                )
                raise MlflowException(message=timeout_message, error_code=BAD_REQUEST)
            else:
                raise e
