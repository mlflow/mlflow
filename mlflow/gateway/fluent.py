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

    Args:
        name (str): The name of the route to fetch.

    Returns:
        Route: An instance of the Route class representing the fetched route.
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

    Args:
        search_filter (str, optional): A string to filter the results of the search. If None, all
                                       routes are returned. Defaults to None.

    Returns:
        List[Route]: A list of Route instances representing the found routes.
    """
    return MlflowGatewayClient().search_routes(search_filter)


@experimental
def query(route: str, data):
    """
    Issues a query request to a configured service through a named route on the Gateway Server.

    This function creates an instance of MlflowGatewayClient and uses it to issue the query to the
    provided route with the provided data.

    Args:
        route (str): The name of the configured route. Route names can be obtained by running
        `mlflow.gateway.search_routes()`
        data: The request payload to be submitted to the route. The exact configuration of
        the expected structure varies based on the route configuration.

    """
    return MlflowGatewayClient().query(route, data)
