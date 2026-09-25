"""
Streamable HTTP MCP endpoint served by the MLflow tracking server (``mlflow server --enable-mcp``).
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from starlette.types import ASGIApp, Receive, Scope, Send

from mlflow.exceptions import MlflowException
from mlflow.mcp.request_context import MCP_REQUEST_USERNAME
from mlflow.mcp.server import collect_category_tools, create_mcp
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, ErrorCode
from mlflow.server.handlers import _get_tracking_store
from mlflow.telemetry.events import McpRunEvent
from mlflow.telemetry.track import _record_event

if TYPE_CHECKING:
    from fastmcp.tools import FunctionTool
    from starlette.applications import Starlette

# Tool categories served remotely. The models and deployments categories are left to the stdio
# server: their tools build Docker images, start local servers and spawn subprocesses on the
# machine running them, which is the tracking server host when served over HTTP.
SERVER_MCP_TOOL_CATEGORIES = ("traces", "scorers", "experiments", "runs")

# Tools of the served categories that are still withheld from the HTTP endpoint because they
# execute work locally rather than against the tracking store:
#
# - ``evaluate_traces`` runs scorers, including LLM judges, in the calling process. Over HTTP that
#   is the tracking server, using the server's own environment and API credentials, so a remote
#   caller could spend those credentials with nothing more than an update grant on an experiment.
#
# These tools remain available on the stdio server (``mlflow mcp run``), where the process
# belongs to the caller.
LOCAL_EXECUTION_TOOLS = frozenset({"evaluate_traces"})


@dataclass(frozen=True)
class McpToolPolicy:
    """
    Authorization hooks applied to every served tool.

    Args:
        authorize: Called before a tool runs with ``(tool_name, username, arguments)``. Denies by
            raising an ``MlflowException`` with the ``PERMISSION_DENIED`` error code.
        validate_coverage: Called once at startup with the served tool names. Raises when a tool
            has no authorization rule, so a new tool cannot be served unguarded.
        is_admin: Whether the user bypasses authorization and result filtering.
        overrides: Replacement implementations for non-admin callers of tools whose results must
            be filtered per caller (unscoped searches).
    """

    authorize: Callable[[str, str | None, dict[str, Any]], None]
    validate_coverage: Callable[[Iterable[str]], None]
    is_admin: Callable[[str | None], bool]
    overrides: Mapping[str, Callable[..., str]] = field(default_factory=dict)


def _authorized_tool(tool: "FunctionTool", policy: McpToolPolicy) -> "FunctionTool":
    from fastmcp.exceptions import ToolError
    from fastmcp.tools import FunctionTool

    original_fn = tool.fn
    override_fn = policy.overrides.get(tool.name)

    def authorized_fn(**kwargs: Any) -> str:
        username = MCP_REQUEST_USERNAME.get()
        if policy.is_admin(username):
            return original_fn(**kwargs)
        try:
            policy.authorize(tool.name, username, kwargs)
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(PERMISSION_DENIED):
                # Deliberately generic: the message must not reveal whether the resource exists.
                raise ToolError("Permission denied") from None
            raise
        fn = override_fn or original_fn
        return fn(**kwargs)

    return FunctionTool(
        fn=authorized_fn,
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    )


class _McpRequestIdentity:
    """
    ASGI wrapper that publishes the authenticated username to tool execution.

    The authentication middleware stores the identity in ``request.state``, which is backed by
    ``scope["state"]``. fastmcp spawns the MCP session task from this call and runs sync tools in
    a worker thread, both of which copy the current context, so the ``ContextVar`` set here is
    visible inside the tool.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        username = scope.get("state", {}).get("username")
        token = MCP_REQUEST_USERNAME.set(username)
        try:
            await self.app(scope, receive, send)
        finally:
            MCP_REQUEST_USERNAME.reset(token)


def create_server_mcp_app(path: str, tool_policy: McpToolPolicy | None = None) -> "Starlette":
    """
    Build the ASGI app that serves the MLflow MCP server over Streamable HTTP.

    Args:
        path: Full request path the endpoint answers on (static prefix included). The returned
            app matches this exact path, so attach it with ``add_route`` rather than ``mount``.
        tool_policy: Authorization hooks applied to every tool; ``None`` serves the tools as-is
            (no authentication configured on the server).

    Returns:
        The MCP app. Attach ``app.state.identity_app`` as the route endpoint (it publishes the
        request identity to tools) and run ``app.lifespan`` inside the server lifespan.
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

    tools = [
        tool
        for tool in collect_category_tools(SERVER_MCP_TOOL_CATEGORIES)
        if tool.name not in LOCAL_EXECUTION_TOOLS
    ]
    if tool_policy is not None:
        tool_policy.validate_coverage(tool.name for tool in tools)
        tools = [_authorized_tool(tool, tool_policy) for tool in tools]

    mcp = create_mcp(tools=tools)
    # Same event as ``mlflow mcp run``; the marker tells the two transports apart.
    _record_event(McpRunEvent, {"context": "server"})
    app = mcp.http_app(path=path, stateless_http=True, transport="streamable-http")
    app.state.identity_app = _McpRequestIdentity(app)
    return app
