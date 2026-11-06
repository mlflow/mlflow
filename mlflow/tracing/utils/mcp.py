import math
import os
import threading
from dataclasses import dataclass
from time import monotonic

from cachetools import LRUCache

from mlflow.entities.mcp_access_endpoint import MCPAccessEndpoint
from mlflow.entities.mcp_server import validate_mcp_server_name
from mlflow.entities.mcp_server_ref import MCPServerRef
from mlflow.entities.mcp_server_version import MCPServerVersion
from mlflow.environment_variables import MLFLOW_MCP_SERVER_CACHE_TTL_SECONDS
from mlflow.exceptions import MlflowException
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.workspace_context import ServerWorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


@dataclass(frozen=True)
class _ResolvedMCPServer:
    name: str
    version: str | None
    url: str | None = None
    transport_type: str | None = None


@dataclass(frozen=True)
class _CacheEntry:
    server: _ResolvedMCPServer
    fetched_at: float
    lookup_started_at: float


_resolution_cache = LRUCache(maxsize=1024)
_cache_lock = threading.Lock()


def _reset_cache_after_fork() -> None:
    global _resolution_cache, _cache_lock

    # A lock held by another thread at fork cannot be released in the child.
    _resolution_cache = LRUCache(maxsize=1024)
    _cache_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_cache_after_fork)


def _validate_workspace(entity: MCPServerVersion | MCPAccessEndpoint, workspace: str) -> None:
    if entity.workspace != workspace:
        raise MlflowException.invalid_parameter_value(
            f"MCP server workspace {entity.workspace!r} does not match "
            f"trace workspace {workspace!r}."
        )


def _version_identity(server: MCPServerVersion, name: str, workspace: str) -> _ResolvedMCPServer:
    _validate_workspace(entity=server, workspace=workspace)
    if server.name != name or server.version is None:
        raise MlflowException.invalid_parameter_value(
            "Resolved MCP server version does not match the requested server."
        )
    ref = MCPServerRef(name=server.name, version=server.version)
    return _ResolvedMCPServer(name=ref.name, version=ref.version)


def _mcp_registry_not_supported_error(store: object) -> MlflowException:
    return MlflowException.invalid_parameter_value(
        "Resolving MCP server aliases and access endpoints requires a tracking store with "
        f"MCP registry support; the configured {type(store).__name__} backend does not "
        "support these lookups."
    )


def resolve_mcp_server(
    server: MCPServerRef | MCPServerVersion | MCPAccessEndpoint,
    tracking_uri: str,
    workspace: str,
    cache_ttl_seconds: float | None,
) -> _ResolvedMCPServer:
    if not isinstance(server, (MCPServerRef, MCPServerVersion, MCPAccessEndpoint)):
        raise MlflowException.invalid_parameter_value(
            "server must be an MCPServerRef, MCPServerVersion, or MCPAccessEndpoint. "
            "Raw names and URLs are not accepted."
        )
    ttl = (
        MLFLOW_MCP_SERVER_CACHE_TTL_SECONDS.get()
        if cache_ttl_seconds is None
        else cache_ttl_seconds
    )
    if not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or ttl < 0:
        raise MlflowException.invalid_parameter_value(
            "MCP server cache TTL must be a finite, non-negative number of seconds."
        )

    if isinstance(server, MCPServerVersion):
        return _version_identity(server=server, name=server.name, workspace=workspace)
    if isinstance(server, MCPAccessEndpoint):
        _validate_workspace(entity=server, workspace=workspace)
        name = server.server_name
        validate_mcp_server_name(name)
        lookup_key = ("endpoint", name, server.id)
    elif server.alias is None:
        return _ResolvedMCPServer(name=server.name, version=server.version)
    else:
        name = server.name
        lookup_key = ("alias", name, server.alias)

    key = (tracking_uri, workspace, *lookup_key)
    resolved = None
    if ttl > 0:
        with _cache_lock:
            if (entry := _resolution_cache.get(key)) and monotonic() - entry.fetched_at < ttl:
                if not isinstance(server, MCPAccessEndpoint) or _endpoint_matches(
                    endpoint=server, resolved=entry.server
                ):
                    resolved = entry.server

    if resolved is None:
        lookup_started_at = monotonic()
        # Use the trace's workspace, even if the caller changed the active workspace since
        # the trace started. This context manager does not mutate the process environment.
        store = _get_store(store_uri=tracking_uri)
        request_workspace = workspace
        if workspace == DEFAULT_WORKSPACE_NAME and not store.supports_workspaces:
            # A workspace header, including "default", is rejected by single-tenant servers.
            request_workspace = None
        with ServerWorkspaceContext(workspace=request_workspace):
            if isinstance(server, MCPAccessEndpoint):
                try:
                    endpoint = store.get_mcp_access_endpoint(
                        server_name=name, endpoint_id=server.id
                    )
                except NotImplementedError as exc:
                    raise _mcp_registry_not_supported_error(store) from exc
                _validate_workspace(entity=endpoint, workspace=workspace)
                if endpoint.id != server.id or endpoint.server_name != name:
                    raise MlflowException.invalid_parameter_value(
                        "Resolved MCP access endpoint does not match the requested endpoint."
                    )
                identity = (
                    _version_identity(
                        server=endpoint.resolved_version, name=name, workspace=workspace
                    )
                    if endpoint.resolved_version is not None
                    else _ResolvedMCPServer(name=name, version=None)
                )
                resolved = _ResolvedMCPServer(
                    name=name,
                    version=identity.version,
                    url=endpoint.url,
                    transport_type=endpoint.transport_type,
                )
            else:
                try:
                    version = store.get_mcp_server_version_by_alias(name=name, alias=server.alias)
                except NotImplementedError as exc:
                    raise _mcp_registry_not_supported_error(store) from exc
                resolved = _version_identity(server=version, name=name, workspace=workspace)
        if isinstance(server, MCPAccessEndpoint):
            _validate_endpoint(endpoint=server, resolved=resolved)
        if ttl > 0:
            with _cache_lock:
                # A slower, older lookup must not replace a newer registry observation.
                cached = _resolution_cache.get(key)
                if cached is None or cached.lookup_started_at <= lookup_started_at:
                    _resolution_cache[key] = _CacheEntry(
                        server=resolved,
                        fetched_at=monotonic(),
                        lookup_started_at=lookup_started_at,
                    )

    if isinstance(server, MCPAccessEndpoint):
        _validate_endpoint(endpoint=server, resolved=resolved)
    return resolved


def _validate_endpoint(endpoint: MCPAccessEndpoint, resolved: _ResolvedMCPServer) -> None:
    if not _endpoint_matches(endpoint=endpoint, resolved=resolved):
        raise MlflowException.invalid_parameter_value(
            "MCP access endpoint URL or transport has changed. Reload the endpoint before linking."
        )


def _endpoint_matches(endpoint: MCPAccessEndpoint, resolved: _ResolvedMCPServer) -> bool:
    return endpoint.url == resolved.url and endpoint.transport_type == resolved.transport_type
