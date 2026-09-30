"""
Per-request identity for MCP tool calls served by the tracking server.

The values are set by the ASGI wrapper around the MCP app from the identity the server's
authentication middleware attached to the request, and read by the tool authorization layer and
by tools whose behavior depends on who is calling. Tools never read a process-wide global: a
``ContextVar`` follows the request through the MCP session task and the worker thread that runs
the tool.
"""

from contextvars import ContextVar

MCP_REQUEST_USERNAME: ContextVar[str | None] = ContextVar("mcp_request_username", default=None)
# True while serving a call through the tracking server's HTTP endpoint, where the process
# belongs to the server rather than to the caller (as it does for ``mlflow mcp run``).
MCP_HTTP_REQUEST: ContextVar[bool] = ContextVar("mcp_http_request", default=False)


def get_mcp_request_username() -> str | None:
    return MCP_REQUEST_USERNAME.get()


def is_mcp_http_request() -> bool:
    return MCP_HTTP_REQUEST.get()
