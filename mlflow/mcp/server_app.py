"""
Streamable HTTP MCP endpoint served by the MLflow tracking server (``mlflow server --enable-mcp``).
"""

import functools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from starlette.types import ASGIApp, Receive, Scope, Send

from mlflow.exceptions import MlflowException
from mlflow.mcp.request_context import MCP_HTTP_REQUEST, MCP_REQUEST_USERNAME
from mlflow.mcp.server import create_mcp, shared_function_tool
from mlflow.mcp.tools import SHARED_TOOLS, SharedTool
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, ErrorCode
from mlflow.server.handlers import _get_tracking_store
from mlflow.telemetry.events import McpRunEvent
from mlflow.telemetry.track import _record_event

if TYPE_CHECKING:
    from starlette.applications import Starlette

# The endpoint serves the typed tools shared with the stdio server. Tools that execute work
# locally stay on the stdio server (``mlflow mcp run``), where the process belongs to the caller:
#
# - ``evaluate_traces`` runs scorers, including LLM judges. Over HTTP that would use the tracking
#   server's own environment and API credentials on behalf of any caller with an update grant.
# - The models and deployments tools build Docker images, start local servers and spawn
#   subprocesses on the machine running them.


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
            be filtered per caller (unscoped searches, scorer listings). An override takes the
            same arguments and returns the same model as the tool it replaces.
        on_success: Called with ``(username, result)`` after a tool succeeds, for every caller.
            Grants the creator MANAGE on what a create tool made, like the REST after-request
            handlers do for the same operations.
    """

    authorize: Callable[[str, str | None, dict[str, Any]], None]
    validate_coverage: Callable[[Iterable[str]], None]
    is_admin: Callable[[str | None], bool]
    overrides: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    on_success: Mapping[str, Callable[[str, Any], None]] = field(default_factory=dict)


def _authorized_fn(tool: SharedTool, policy: McpToolPolicy) -> Callable[..., Any]:
    from fastmcp.exceptions import ToolError

    override_fn = policy.overrides.get(tool.name)
    on_success = policy.on_success.get(tool.name)

    def run(username: str | None, **kwargs: Any) -> Any:
        if policy.is_admin(username):
            return tool.fn(**kwargs)
        try:
            policy.authorize(tool.name, username, kwargs)
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(PERMISSION_DENIED):
                # Deliberately generic: the message must not reveal whether the resource exists.
                raise ToolError("Permission denied") from None
            raise
        return (override_fn or tool.fn)(**kwargs)

    # ``functools.wraps`` carries the typed signature, annotations and docstring over, so FastMCP
    # validates the arguments against the tool's own schema before the authorization check runs.
    @functools.wraps(tool.fn)
    def authorized_fn(**kwargs: Any) -> Any:
        username = MCP_REQUEST_USERNAME.get()
        result = run(username, **kwargs)
        if on_success is not None:
            on_success(username, result)
        return result

    return authorized_fn


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
        username_token = MCP_REQUEST_USERNAME.set(username)
        http_token = MCP_HTTP_REQUEST.set(True)
        try:
            await self.app(scope, receive, send)
        finally:
            MCP_HTTP_REQUEST.reset(http_token)
            MCP_REQUEST_USERNAME.reset(username_token)


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

    # The tools resolve their store from the configured tracking URI, like any MLflow client.
    # Initializing the server's tracking store first points that URI at the backend store, so the
    # tools read and write the same store the REST API serves rather than a local ./mlruns.
    _get_tracking_store()

    if tool_policy is None:
        tools = [shared_function_tool(tool) for tool in SHARED_TOOLS]
    else:
        tool_policy.validate_coverage(tool.name for tool in SHARED_TOOLS)
        tools = [
            shared_function_tool(tool, _authorized_fn(tool, tool_policy)) for tool in SHARED_TOOLS
        ]

    mcp = create_mcp(tools=tools)
    # Same event as ``mlflow mcp run``; the marker tells the two transports apart.
    _record_event(McpRunEvent, {"context": "server"})
    app = mcp.http_app(path=path, stateless_http=True, transport="streamable-http")
    app.state.identity_app = _McpRequestIdentity(app)
    return app
