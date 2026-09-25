"""
Streamable HTTP MCP endpoint served by the MLflow tracking server (``mlflow server --enable-mcp``).
"""

from typing import TYPE_CHECKING

from mlflow.exceptions import MlflowException
from mlflow.mcp.server import create_mcp
from mlflow.server.handlers import _get_tracking_store

if TYPE_CHECKING:
    from starlette.applications import Starlette

# Only the GenAI tools are exposed remotely. The models and deployments tools build Docker images,
# start local servers and spawn subprocesses on the machine running them, which is the tracking
# server host when served over HTTP.
SERVER_MCP_TOOL_CATEGORIES = ("traces", "scorers", "experiments", "runs")


def create_server_mcp_app(path: str) -> "Starlette":
    """
    Build the ASGI app that serves the MLflow MCP server over Streamable HTTP.

    Args:
        path: Full request path the endpoint answers on (static prefix included). The returned
            app matches this exact path, so attach it with ``add_route`` rather than ``mount``.
    """
    try:
        import fastmcp  # noqa: F401
    except ImportError as e:
        raise MlflowException(
            "The `fastmcp` package is required to serve the MCP endpoint. Install it with "
            "`pip install fastmcp` or start the server without `--enable-mcp`."
        ) from e

    # The MCP tools go through the MLflow client API. Resolving the server's tracking store first
    # points the in-process tracking URI at the backend store, so the tools read and write the
    # same store the REST API serves rather than a local ./mlruns directory.
    _get_tracking_store()

    mcp = create_mcp(categories=SERVER_MCP_TOOL_CATEGORIES)
    return mcp.http_app(path=path, stateless_http=True, transport="streamable-http")
