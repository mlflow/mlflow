"""
Per-request identity for MCP tool calls served by the tracking server.

The value is set by the ASGI wrapper around the MCP app from the identity the server's
authentication middleware attached to the request, and read by the tool authorization layer.
Tools never read a process-wide global: a ``ContextVar`` follows the request through the MCP
session task and the worker thread that runs the tool.
"""

from contextvars import ContextVar

MCP_REQUEST_USERNAME: ContextVar[str | None] = ContextVar("mcp_request_username", default=None)


def get_mcp_request_username() -> str | None:
    return MCP_REQUEST_USERNAME.get()
