import json
import logging
from typing import Any, Dict, List, Optional

import requests.exceptions

from mlflow import MlflowException
from mlflow.gateway.config import LimitsConfig, Route
from mlflow.gateway.constants import (
    MLFLOW_GATEWAY_CLIENT_QUERY_RETRY_CODES,
    MLFLOW_GATEWAY_CLIENT_QUERY_TIMEOUT_SECONDS,
    MLFLOW_GATEWAY_CRUD_ROUTE_BASE,
    MLFLOW_GATEWAY_LIMITS_BASE,
    MLFLOW_GATEWAY_ROUTE_BASE,
    MLFLOW_QUERY_SUFFIX,
)
from mlflow.gateway.utils import (
    assemble_uri_path,
    gateway_deprecated,
    get_gateway_uri,
    resolve_route_url,
)
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.store.entities.paged_list import PagedList
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request
from mlflow.utils.uri import get_uri_scheme

_logger = logging.getLogger(__name__)


@gateway_deprecated
class MlflowGatewayClient:
    """
    Client for interacting with the MLflow Gateway API.

    Args:
        gateway_uri: Optional URI of the gateway. If not provided, attempts to resolve from
            first the stored result of `set_gateway_uri()`, then the  environment variable
            `MLFLOW_GATEWAY_URI`.
    """

    def __init__(self, gateway_uri: Optional[str] = None):
        self._gateway_uri = gateway_uri or get_gateway_uri()

    def _is_databricks_host(self) -> bool:
        return (
            self._gateway_uri == "databricks" or get_uri_scheme(self._gateway_uri) == "databricks"
        )

    @property
    def _host_creds(self):
        """
        NB: When `MlflowGatewayClient` is used as an instance variable in a custom pyfunc model, it
        is pickled in the environment where the custom pyfunc model is defined (e.g. a notebook).
        When the model is moved to a different environment, e.g. model serving, new credentials
        need to be resolved from within the new environment. Accordingly, we re-resolve host
        credentials every time a request is made.
        """
        if self._is_databricks_host():
            return get_databricks_host_creds(self._gateway_uri)
        else:
            return get_default_host_creds(self._gateway_uri)

    @property
    def gateway_uri(self):
        """
        Get the current value for the URI of the MLflow Gateway.

        Returns:
            The gateway URI.
        """
        return self._gateway_uri

    def _call_endpoint(self, method: str, route: str, json_body: Optional[str] = None):
        """
        Call a specific endpoint on the Gateway API.

        Args:
            method: The HTTP method to use.
            route: The API route to call.
            json_body: Optional JSON body to include in the request.

        Returns:
            The server's response.

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
            raise_on_status=False,
            **call_kwargs,
        )
        augmented_raise_for_status(response)
        return response

    @gateway_deprecated
    def get_route(self, name: str):
        """
        Get a specific query route from the gateway. The routes that are available to retrieve
        are only those that have been configured through the MLflow Gateway Server configuration
        file (set during server start or through server update commands).

        Args:
            name: The name of the route.

        Returns:
            The returned data structure is a serialized representation of the `Route` data
            structure, giving information about the name, type, and model details (model name
            and provider) for the requested route endpoint.

        """
        route = assemble_uri_path([MLFLOW_GATEWAY_CRUD_ROUTE_BASE, name])
        response = self._call_endpoint("GET", route).json()
        response["route_url"] = resolve_route_url(self._gateway_uri, response["route_url"])

        return Route(**response)

    @gateway_deprecated
    def search_routes(self, page_token: Optional[str] = None) -> PagedList[Route]:
        """
        Search for routes in the Gateway.

        Args:
            page_token: Token specifying the next page of results. It should be obtained from
                a prior ``search_routes()`` call.

        Returns:
            Returns a list of all configured and initialized `Route` data for the MLflow
            Gateway Server. The return will be a list of dictionaries that detail the name, type,
            and model details of each active route endpoint.

        """
        request_parameters = {"page_token": page_token} if page_token is not None else None
        response_json = self._call_endpoint(
            "GET", MLFLOW_GATEWAY_CRUD_ROUTE_BASE, json_body=json.dumps(request_parameters)
        ).json()
        routes = [
            Route(
                **{
                    **resp,
                    "route_url": resolve_route_url(
                        self._gateway_uri,
                        resp["route_url"],
                    ),
                }
            )
            for resp in response_json.get("routes", [])
        ]
        next_page_token = response_json.get("next_page_token")
        return PagedList(routes, next_page_token)

    @gateway_deprecated
    def create_route(
        self, name: str, route_type: Optional[str] = None, model: Optional[Dict[str, Any]] = None
    ) -> Route:
        """
        Create a new route in the Gateway.

        .. warning::

            This API is **only available** when running within Databricks. When running elsewhere,
            route configuration is handled via updates to the route configuration YAML file that
            is specified during Gateway server start.

        Args:
            name: The name of the route. This parameter is required for all routes.
            route_type: The type of the route (e.g., 'llm/v1/chat', 'llm/v1/completions',
                'llm/v1/embeddings'). This parameter is required for routes that are not managed by
                Databricks (the provider isn't 'databricks').
            model: A dictionary representing the model details to be associated with the route.
                This parameter is required for all routes. This dictionary should define:

                    - The model name (e.g., "gpt-3.5-turbo")
                    - The provider (e.g., "openai", "anthropic")
                    - The configuration for the model used in the route

        Returns:
            A serialized representation of the `Route` data structure,
            providing information about the name, type, and model details for the
            newly created route endpoint.

        Raises:
            mlflow.MlflowException: If the function is not running within Databricks.

        .. note::

            See the official Databricks documentation for MLflow Gateway for examples of supported
            model configurations and how to dynamically create new routes within Databricks.

        Example usage from within Databricks:

        .. code-block:: python

            from mlflow.gateway import MlflowGatewayClient

            gateway_client = MlflowGatewayClient("databricks")

            openai_api_key = ...

            new_route = gateway_client.create_route(
                name="my-route",
                route_type="llm/v1/completions",
                model={
                    "name": "question-answering-bot",
                    "provider": "openai",
                    "openai_config": {
                        "openai_api_key": openai_api_key,
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

    @gateway_deprecated
    def delete_route(self, name: str) -> None:
        """
        Delete an existing route in the Gateway.

        .. warning::

            This API is **only available** when running within Databricks. When running elsewhere,
            route deletion is handled by removing the corresponding entry from the route
            configuration YAML file that is specified during Gateway server start.

        Args:
            name: The name of the route to delete.

        Raises:
            mlflow.MlflowException: If the function is not running within Databricks.

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

    @gateway_deprecated
    def query(self, route: str, data: Dict[str, Any]):
        """
        Submit a query to a configured provider route.

        Args:
            route: The name of the route to submit the query to.
            data: The data to send in the query. A dictionary representing the per-route
                specific structure required for a given provider.

                For chat, the structure should be:

                .. code-block:: python

                    from mlflow.gateway import MlflowGatewayClient

                    gateway_client = MlflowGatewayClient("http://my.gateway:8888")

                    response = gateway_client.query(
                        "my-chat-route",
                        {
                            "messages": [
                                {"role": "user", "content": "Tell me a joke about rabbits"},
                            ]
                        },
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

                Additional parameters that are valid for a given provider and route configuration
                can be included with the request as shown below, using an openai completions route
                request as an example:

                .. code-block:: python

                    from mlflow.gateway import MlflowGatewayClient

                    gateway_client = MlflowGatewayClient("http://my.gateway:8888")

                    response = gateway_client.query(
                        "my-completions-route",
                        {
                            "prompt": "Give me an example of a properly formatted pytest unit test",
                            "temperature": 0.3,
                            "max_tokens": 500,
                        },
                    )

        Returns:
            The route's response as a dictionary, standardized to the route type.

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

    @gateway_deprecated
    def set_limits(self, route: str, limits: List[Dict[str, Any]]) -> LimitsConfig:
        """
        Set limits on an existing route in the Gateway.

        .. warning::

            This API is **only available** when running within Databricks.

        Args:
            route: The name of the route to set limits on.
            limits: Limits (Array of dictionary) to set on the route. Each limit is defined by a
                dictionary representing the limit details to be associated with the route. This
                dictionary should define:

                - renewal_period: a string representing the length of the window to enforce limit
                  on (only supports "minute" for now).
                - calls: a non-negative integer representing the number of calls allowed per
                  renewal_period (e.g., 10, 0, 55).
                - key: an optional string represents per route limit or per user limit ("user" for
                  per user limit, "route" for per route limit, if not supplied, default to per
                  route limit).

        Returns:
            The returned data structure is a serialized representation of the `Limit`
            data structure, giving information about the renewal_period, key, and calls.

        Example usage:

        .. code-block:: python

            from mlflow.gateway import MlflowGatewayClient

            gateway_client = MlflowGatewayClient("databricks")

            gateway_client.set_limits(
                "my-new-route", [{"key": "user", "renewal_period": "minute", "calls": 50}]
            )
        """
        payload = {
            "route": route,
            "limits": limits,
        }

        response = self._call_endpoint(
            "POST", MLFLOW_GATEWAY_LIMITS_BASE, json.dumps(payload)
        ).json()
        return LimitsConfig(**response)

    @gateway_deprecated
    def get_limits(self, route: str) -> LimitsConfig:
        """
        Get limits of an existing route in the Gateway.

        .. warning::

            This API is **only available** when connected to a Databricks-hosted AI Gateway.

        Args:
            route: The name of the route to get limits of.

        Returns:
            The returned data structure is a serialized representation of the `Limit` data
            structure, giving information about the renewal_period, key, and calls.

        Example usage:

        .. code-block:: python

            from mlflow.gateway import MlflowGatewayClient

            gateway_client = MlflowGatewayClient("databricks")

            gateway_client.get_limits("my-new-route")
        """
        if not route:
            raise MlflowException("A non-empty string is required for the route.", BAD_REQUEST)
        route_uri = assemble_uri_path([MLFLOW_GATEWAY_LIMITS_BASE, route])
        response = self._call_endpoint("GET", route_uri).json()
        return LimitsConfig(**response)
