import logging
from typing import Any, Dict, List, Optional

from mlflow.gateway.client import MlflowGatewayClient
from mlflow.gateway.config import LimitsConfig, Route
from mlflow.gateway.constants import MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE
from mlflow.utils import get_results_from_paginated_fn
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@experimental
def get_route(name: str) -> Route:
    """
    Retrieves a specific route from the MLflow Gateway service.

    This function creates an instance of MlflowGatewayClient and uses it to fetch a route by its
    name from the Gateway service.

    :param name: The name of the route to fetch.
    :return: An instance of the Route class representing the fetched route.
    """
    return MlflowGatewayClient().get_route(name)


@experimental
def search_routes() -> List[Route]:
    """
    Searches for routes in the MLflow Gateway service.

    This function creates an instance of MlflowGatewayClient and uses it to fetch a list of routes
    from the Gateway service.

    :return: A list of Route instances.
    """

    def pagination_wrapper_func(_, next_page_token):
        return MlflowGatewayClient().search_routes(page_token=next_page_token)

    return get_results_from_paginated_fn(
        paginated_fn=pagination_wrapper_func,
        max_results_per_page=MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE,
        max_results=None,
    )


@experimental
def create_route(
    name: str, route_type: Optional[str] = None, model: Optional[Dict[str, Any]] = None
) -> Route:
    """
    Create a new route in the Gateway.

    .. warning::

        This API is ``only available`` when running within Databricks. When running elsewhere,
        route configuration is handled via updates to the route configuration YAML file that
        is specified during Gateway server start.

    :param name: The name of the route. This parameter is required for all routes.
    :param route_type: The type of the route (e.g., 'llm/v1/chat', 'llm/v1/completions',
                       'llm/v1/embeddings'). This parameter is required for routes that are
                       not managed by Databricks (the provider isn't 'databricks').
    :param model: A dictionary representing the model details to be associated with the route.
                  This parameter is required for all routes. This dictionary should define:

                  - The model name (e.g., "gpt-3.5-turbo")
                  - The provider (e.g., "openai", "anthropic")
                  - The configuration for the model used in the route

    :return: A serialized representation of the `Route` data structure,
             providing information about the name, type, and model details for the
             newly created route endpoint.

    .. note::

        See the official Databricks documentation for MLflow Gateway for examples of supported
        model configurations and how to dynamically create new routes within Databricks.


    Example usage from within Databricks:

    .. code-block:: python

        from mlflow.gateway import set_gateway_uri, create_route

        set_gateway_uri(gateway_uri="databricks")

        openai_api_key = ...

        create_route(
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
    return MlflowGatewayClient().create_route(name, route_type, model)


@experimental
def delete_route(name: str) -> None:
    """
    Delete an existing route in the Gateway.

    .. warning::

        This API is **only available** when running within Databricks. When running elsewhere,
        route deletion is handled by removing the corresponding entry from the route
        configuration YAML file that is specified during Gateway server start.

    :param name: The name of the route to delete.

    Example usage from within Databricks:

    .. code-block:: python

        from mlflow.gateway import set_gateway_uri, delete_route

        set_gateway_uri(gateway_uri="databricks")

        delete_route("my-new-route")

    """
    MlflowGatewayClient().delete_route(name)


@experimental
def set_limits(route: str, limits: List[Dict[str, Any]]) -> LimitsConfig:
    """
    Set limits on an existing route in the Gateway.

    .. warning::

        This API is **only available** when running within Databricks.

    :param route: The name of the route to set limits on.
    :param limits: Limits to set on the route.

    Example usage from within Databricks:

    .. code-block:: python

        from mlflow.gateway import set_gateway_uri, set_limits

        set_gateway_uri(gateway_uri="databricks")

        set_limits("my-new-route", [{"key": "user", "renewal_period": "minute", "calls": 50}])

    """
    return MlflowGatewayClient().set_limits(route=route, limits=limits)


@experimental
def get_limits(route: str) -> LimitsConfig:
    """
    Get limits of an existing route in the Gateway.

    .. warning::

        This API is **only available** when connected to a Databricks-hosted AI Gateway.

    :param route: The name of the route to get limits of.

    Example usage from within Databricks:

    .. code-block:: python

        from mlflow.gateway import set_gateway_uri, get_limits

        set_gateway_uri(gateway_uri="databricks")

        get_limits("my-new-route")

    """
    return MlflowGatewayClient().get_limits(route=route)


@experimental
def query(route: str, data):
    """
    Issues a query request to a configured service through a named route on the Gateway Server.
    This function will interface with a configured route name (examples below) and return the
    response from the provider in a standardized format.

    :param route: The name of the configured route. Route names can be obtained by running
                  `mlflow.gateway.search_routes()`
    :param data: The request payload to be submitted to the route. The exact configuration of
                 the expected structure varies based on the route configuration.
    :return: The response from the configured route endpoint provider in a standardized format.

    Chat example:

    .. code-block:: python

        from mlflow.gateway import query, set_gateway_uri

        set_gateway_uri(gateway_uri="http://my.gateway:9000")
        response = query(
            "my_chat_route",
            {"messages": [{"role": "user", "content": "What is the best day of the week?"}]},
        )

    Completions example:

    .. code-block:: python

        from mlflow.gateway import query, set_gateway_uri

        set_gateway_uri(gateway_uri="http://my.gateway:9000")
        response = query("a_completions_route", {"prompt": "Where do we go from"})

    Embeddings example:

    .. code-block:: python

        from mlflow.gateway import query, set_gateway_uri

        set_gateway_uri(gateway_uri="http://my.gateway:9000")
        response = query(
            "embeddings_route", {"text": ["I like spaghetti", "and sushi", "but not together"]}
        )

    Additional parameters that are valid for a given provider and route configuration can be
    included with the request as shown below, using an openai completions route request as
    an example:

    .. code-block:: python

        from mlflow.gateway import query, set_gateway_uri

        set_gateway_uri(gateway_uri="http://my.gateway:9000")
        response = query(
            "a_completions_route",
            {
                "prompt": "Give me an example of a properly formatted pytest unit test",
                "temperature": 0.6,
                "max_tokens": 1000,
            },
        )

    """
    return MlflowGatewayClient().query(route, data)
