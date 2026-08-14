"""Client-side MCP tool discovery helpers."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Mapping

from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPTool
from mlflow.environment_variables import MLFLOW_ENABLE_MCP_TOOL_DISCOVERY
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.mcp_server_registry.abstract_mixin import NOT_SET

# Bound create-time discovery so a hung remote cannot stall registration.
# On timeout, discovery is skipped and create continues with tools=None.
DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS = 10.0
# Client HTTP timeouts are slightly looser so asyncio.wait_for owns the
# user-facing deadline and the clearer "Timed out discovering" error path.
_CLIENT_TIMEOUT_SLACK_SECONDS = 1.0

_logger = logging.getLogger(__name__)


def _run_coro_sync(coro: Any, timeout: float | None = None) -> Any:
    """Run an async coroutine from a synchronous caller.

    Uses ``asyncio.run`` directly unless the caller is already inside a running
    event loop, in which case a daemon worker thread is used so the sync API
    can still wait for completion.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    outcome: dict[str, Any] = {}

    def _worker() -> None:
        try:
            outcome["value"] = asyncio.run(coro)
        except Exception as e:
            outcome["error"] = e

    thread = threading.Thread(target=_worker, name="mcp_tool_discovery", daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        raise MlflowException.invalid_parameter_value(
            f"Timed out discovering MCP tools after {timeout:g}s"
            if timeout is not None
            else "Timed out discovering MCP tools"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _require_fastmcp():
    try:
        from fastmcp import Client
        from fastmcp.client.transports import SSETransport, StreamableHttpTransport
    except ImportError as e:
        raise MlflowException.invalid_parameter_value(
            "MCP tool discovery requires the optional MCP dependencies. "
            "Install them with: pip install 'mlflow[mcp]'"
        ) from e
    return Client, StreamableHttpTransport, SSETransport


def _tool_from_sdk(tool: Any) -> MCPTool:
    # fastmcp Client.list_tools() returns Pydantic Tool models.
    data = tool.model_dump(by_alias=True, exclude_none=True)
    # Drop MCP protocol-only fields that MCPTool does not model.
    data.pop("meta", None)
    return MCPTool.from_dict(data)


def _build_transport(
    url: str,
    transport_type: MCPRemoteTransportType,
    headers: Mapping[str, str] | None,
    StreamableHttpTransport,
    SSETransport,
):
    header_dict = dict(headers) if headers else None
    if transport_type == MCPRemoteTransportType.STREAMABLE_HTTP:
        return StreamableHttpTransport(url=url, headers=header_dict)
    if transport_type == MCPRemoteTransportType.SSE:
        return SSETransport(url=url, headers=header_dict)
    raise MlflowException.invalid_parameter_value(
        f"Unsupported MCP transport for tool discovery: {transport_type!r}"
    )


async def _alist_tools(
    url: str,
    transport_type: MCPRemoteTransportType,
    headers: Mapping[str, str] | None,
    timeout_seconds: float,
) -> list[MCPTool]:
    Client, StreamableHttpTransport, SSETransport = _require_fastmcp()
    transport = _build_transport(
        url, transport_type, headers, StreamableHttpTransport, SSETransport
    )
    # wait_for owns the advertised deadline; client budget is slightly larger
    # so a hung remote surfaces as "Timed out discovering" rather than a raw
    # client ReadTimeout.
    client_timeout = timeout_seconds + _CLIENT_TIMEOUT_SLACK_SECONDS

    async def _list() -> list[Any]:
        async with Client(
            transport,
            timeout=client_timeout,
            init_timeout=client_timeout,
        ) as client:
            return await client.list_tools()

    try:
        sdk_tools = await asyncio.wait_for(_list(), timeout=timeout_seconds)
    except asyncio.TimeoutError as e:
        # On 3.11+ asyncio.TimeoutError is an alias of TimeoutError; catch the
        # asyncio form so 3.10 also maps wait_for timeouts correctly.
        raise MlflowException.invalid_parameter_value(
            f"Timed out discovering MCP tools from {url!r} after {timeout_seconds:g}s"
        ) from e
    return [_tool_from_sdk(t) for t in sdk_tools]


def discover_mcp_tools(
    url: str,
    transport_type: MCPRemoteTransportType = MCPRemoteTransportType.STREAMABLE_HTTP,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS,
) -> list[MCPTool]:
    """List tools from a deployed MCP server endpoint."""
    if timeout <= 0:
        raise MlflowException.invalid_parameter_value(f"timeout must be positive, got {timeout!r}")
    try:
        return _run_coro_sync(
            _alist_tools(url, transport_type, headers, timeout),
            timeout=timeout + _CLIENT_TIMEOUT_SLACK_SECONDS,
        )
    except MlflowException:
        raise
    except Exception as e:
        raise MlflowException.invalid_parameter_value(
            f"Failed to discover MCP tools from {url!r}: {e}"
        ) from e


def _first_discovery_remote(
    server_json: dict[str, Any],
) -> tuple[str, MCPRemoteTransportType] | None:
    """Return the first remotes[] entry usable for tool discovery."""
    remotes = server_json.get("remotes") or []
    if not isinstance(remotes, list):
        raise MlflowException.invalid_parameter_value(
            "Invalid server_json.remotes. Expected a list of remote objects."
        )

    for remote in remotes:
        if not isinstance(remote, dict):
            _logger.info(
                "Skipping non-object MCP remotes entry for tool discovery; trying next if any"
            )
            continue
        url = remote.get("url")
        if url is None:
            continue
        if not isinstance(url, str) or not url.strip():
            _logger.info(
                "Skipping MCP remote with blank/non-string url for tool discovery; "
                "trying next if any"
            )
            continue
        transport_str = "streamable-http" if remote.get("type") is None else remote.get("type")
        try:
            transport = MCPRemoteTransportType(transport_str)
        except ValueError:
            _logger.info(
                "Skipping MCP remote with unsupported transport %r for tool discovery; "
                "trying next if any",
                transport_str,
            )
            continue

        return url.strip(), transport
    return None


def discover_tools_for_server_json(
    server_json: dict[str, Any],
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS,
) -> list[MCPTool]:
    """Discover tools from the first usable remote in a ``server.json`` payload."""
    remote = _first_discovery_remote(server_json)
    if remote is None:
        raise MlflowException.invalid_parameter_value(
            "No usable MCP remote found in server_json.remotes for tool discovery."
        )
    url, transport = remote
    return discover_mcp_tools(
        url=url,
        transport_type=transport,
        headers=headers,
        timeout=timeout,
    )


def resolve_tools_for_create(
    server_json: dict[str, Any],
    tools: list[MCPTool] | None,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS,
) -> list[MCPTool] | None:
    """Resolve ``tools`` for MCP server version create.

    * ``NOT_SET`` (Python default / JSON field omitted): best-effort auto-discover
      from the first usable ``server_json.remotes[]`` URL. No usable remote,
      selection/scrape failure, or timeout -> ``None`` (create continues). Live
      scrape does not failover to later remotes.
    * ``None`` (explicit JSON null): store no tools; do not discover.
    * ``[]`` / a list: store as-is; do not discover.
    """
    if tools is not NOT_SET:
        return tools

    if not MLFLOW_ENABLE_MCP_TOOL_DISCOVERY.get():
        return None

    try:
        return discover_tools_for_server_json(server_json, headers=headers, timeout=timeout)
    except Exception as e:
        _logger.warning(
            "MCP tool discovery failed; creating version with tools=None: %s",
            e,
        )
        return None
