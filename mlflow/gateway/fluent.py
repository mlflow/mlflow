import logging
from typing import Optional, List

from mlflow.gateway.client import MlflowGatewayClient
from mlflow.gateway.config import Route
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
def search_routes(search_filter: Optional[str] = None) -> List[Route]:
    """
    Searches for routes in the MLflow Gateway service.

    This function creates an instance of MlflowGatewayClient and uses it to fetch a list of routes
    from the Gateway service, optionally filtered by a search string.

    .. note::
        Search is currently not implemented. Providing a `search_filter` term will raise an
        exception. Leave the arguments empty to get a listing of all configured routes.

    :param search_filter: A string to filter the results of the search. If None, all
                          routes are returned. Defaults to None.
    :return: A list of Route instances representing the found routes.
    """
    return MlflowGatewayClient().search_routes(search_filter)


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
    """
    return MlflowGatewayClient().query(route, data)
