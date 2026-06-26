from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPStatus, MCPTool
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.mcp_server_registry.abstract_mixin import NOT_SET, MCPIcon
from mlflow.tracking.client import MlflowClient
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.uri import get_uri_scheme, is_local_uri

if TYPE_CHECKING:
    from enum import Enum

    from mlflow.entities.mcp_access_binding import MCPAccessBinding
    from mlflow.entities.mcp_server import MCPServer
    from mlflow.entities.mcp_server_version import MCPServerVersion


def _parse_enum(value: Any, enum_cls: type[Enum], param_name: str) -> Any:
    if value is NOT_SET or value is None:
        return value
    try:
        return enum_cls(value)
    except ValueError:
        valid = ", ".join(repr(e.value) for e in enum_cls)
        raise MlflowException.invalid_parameter_value(
            f"Invalid {param_name} {value!r}. Valid values are: {valid}"
        ) from None


def register_mcp_server(
    server_json: dict[str, Any],
    display_name: str | None = None,
    source: str | None = None,
    status: Literal["draft", "active", "deprecated", "deleted"] = "draft",
    tools: list[MCPTool] | None = None,
    create_access_bindings_from_remotes: bool = False,
) -> MCPServerVersion:
    """Register an MCP server from a ``server_json`` payload.

    If the parent ``MCPServer`` does not exist, it is created automatically.
    If the version already exists, a ``MlflowException`` is raised.

    Args:
        server_json: The canonical MCP ``server.json`` payload. Must contain ``name``
            and ``version`` at the top level.
        display_name: Human-readable label for the version.
        source: Provenance URI (e.g., a git repository URL).
        status: Initial status (default ``"draft"``).
        tools: Declared tool definitions for this version.
        create_access_bindings_from_remotes: When ``True``, create one direct-access
            binding per ``remotes[]`` entry in ``server_json``.

    Returns:
        The created :py:class:`MCPServerVersion <mlflow.entities.MCPServerVersion>`.

    Example:

    .. code-block:: python

        import mlflow

        version = mlflow.genai.register_mcp_server(
            server_json={
                "name": "io.github.anthropic/brave-search",
                "version": "1.0.0",
                "description": "Brave Search MCP server",
            },
        )
        assert version.status == MCPStatus.DRAFT
    """
    client = MlflowClient()
    parsed_status = _parse_enum(status, MCPStatus, "status")

    should_create_bindings = create_access_bindings_from_remotes and parsed_status in (
        MCPStatus.ACTIVE,
        MCPStatus.DEPRECATED,
    )

    validated_remotes = []
    if should_create_bindings:
        for remote in server_json.get("remotes", []):
            url = remote.get("url")
            if not url:
                continue
            transport_str = remote.get("type", "streamable-http")
            transport = _parse_enum(transport_str, MCPRemoteTransportType, "transport_type")
            validated_remotes.append((url, transport))

    version = client.create_mcp_server_version(
        server_json=server_json,
        display_name=display_name,
        source=source,
        status=parsed_status,
        tools=tools,
    )

    for url, transport in validated_remotes:
        client.create_mcp_access_binding(
            server_name=version.name,
            endpoint_url=url,
            transport_type=transport,
            server_version=version.version,
        )

    return version


_MAX_SERVER_JSON_BYTES = 10 * 1024 * 1024  # 10 MiB


def _sanitize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunparse((parsed.scheme, host, parsed.path, "", "", ""))


def _read_server_json_local(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise MlflowException.invalid_parameter_value(f"Local server.json not found: {path}")
    data = file_path.read_bytes()
    if len(data) > _MAX_SERVER_JSON_BYTES:
        raise MlflowException.invalid_parameter_value(
            f"server.json exceeds {_MAX_SERVER_JSON_BYTES} byte limit"
        )
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise MlflowException.invalid_parameter_value(
            f"File {path} does not contain valid JSON: {e}"
        ) from e


def _read_server_json_remote(url: str) -> dict[str, Any]:
    safe_url = _sanitize_url(url)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read(_MAX_SERVER_JSON_BYTES + 1)
    except urllib.error.HTTPError as e:
        raise MlflowException.invalid_parameter_value(
            f"Failed to fetch server.json from {safe_url}: HTTP {e.code}"
        ) from e
    except urllib.error.URLError as e:
        raise MlflowException.invalid_parameter_value(
            f"Failed to fetch server.json from {safe_url}: {e.reason}"
        ) from e

    if len(data) > _MAX_SERVER_JSON_BYTES:
        raise MlflowException.invalid_parameter_value(
            f"server.json response exceeds {_MAX_SERVER_JSON_BYTES} byte limit"
        )
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise MlflowException.invalid_parameter_value(
            f"URL {safe_url} did not return valid JSON: {e}"
        ) from e


def register_mcp_server_from_url(
    url: str,
    display_name: str | None = None,
    source: str | None = None,
    status: Literal["draft", "active", "deprecated", "deleted"] = "draft",
    tools: list[MCPTool] | None = None,
    create_access_bindings_from_remotes: bool = False,
) -> MCPServerVersion:
    """Fetch a ``server.json`` from a URL or local file path and register the MCP server.

    Args:
        url: HTTP/HTTPS URL or local file path (absolute path or ``file://`` URI)
            pointing to a ``server.json`` document.
        display_name: Human-readable label for the version.
        source: Provenance URI; defaults to ``url`` when not provided.
        status: Initial status (default ``"draft"``).
        tools: Declared tool definitions for this version.
        create_access_bindings_from_remotes: When ``True``, create one direct-access
            binding per ``remotes[]`` entry in ``server_json``.

    Returns:
        The created :py:class:`MCPServerVersion <mlflow.entities.MCPServerVersion>`.
    """
    if is_local_uri(url):
        server_json = _read_server_json_local(local_file_uri_to_path(url))
    else:
        scheme = get_uri_scheme(url)
        match scheme:
            case "http" | "https":
                server_json = _read_server_json_remote(url)
            case _:
                raise MlflowException.invalid_parameter_value(
                    f"URL must use http, https, or file scheme, got: {scheme!r}"
                )

    return register_mcp_server(
        server_json=server_json,
        display_name=display_name,
        source=source or _sanitize_url(url),
        status=status,
        tools=tools,
        create_access_bindings_from_remotes=create_access_bindings_from_remotes,
    )


# --- MCPServer CRUD ---


def get_mcp_server(name: str) -> MCPServer:
    """Get an MCP server by name.

    Args:
        name: Server name.

    Returns:
        The :py:class:`MCPServer <mlflow.entities.MCPServer>` with tags, aliases, and
        access bindings populated.
    """
    return MlflowClient().get_mcp_server(name=name)


def search_mcp_servers(
    filter_string: str | None = None,
    max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
    order_by: list[str] | None = None,
    page_token: str | None = None,
) -> PagedList[MCPServer]:
    """Search MCP servers with optional filtering and pagination.

    Args:
        filter_string: SQL-like filter expression (e.g., ``"status = 'active'"``,
            ``"tags.team = 'platform'"``, ``"has_access_bindings = 'true'"``).
        max_results: Maximum number of results to return.
        order_by: List of columns to order by.
        page_token: Token for retrieving the next page of results.

    Returns:
        A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
        :py:class:`MCPServer <mlflow.entities.MCPServer>` objects.

    Example:

    .. code-block:: python

        import mlflow

        servers = mlflow.genai.search_mcp_servers(filter_string="status = 'active'")
        for server in servers:
            print(server.name, server.status)
    """
    return MlflowClient().search_mcp_servers(
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )


def update_mcp_server(
    name: str,
    display_name: str | None = NOT_SET,
    description: str | None = NOT_SET,
    icons: list[MCPIcon] | None = NOT_SET,
) -> MCPServer:
    """Update mutable fields of an MCP server.

    Only fields that are explicitly passed are updated; omitted fields are left
    unchanged. Pass ``None`` to clear a field.

    Args:
        name: Server name.
        display_name: New human-readable label. Pass ``None`` to clear.
        description: New description. Pass ``None`` to clear.
        icons: New icon variants. Pass ``None`` to clear.

    Returns:
        The updated :py:class:`MCPServer <mlflow.entities.MCPServer>`.
    """
    return MlflowClient().update_mcp_server(
        name=name,
        display_name=display_name,
        description=description,
        icons=icons,
    )


def delete_mcp_server(name: str) -> None:
    """Delete an MCP server and all its child entities.

    Args:
        name: Server name.
    """
    MlflowClient().delete_mcp_server(name=name)


# --- MCPServerVersion CRUD ---


def get_mcp_server_version(name: str, version: str) -> MCPServerVersion:
    return MlflowClient().get_mcp_server_version(name=name, version=version)


def get_mcp_server_version_by_alias(name: str, alias: str) -> MCPServerVersion:
    """Resolve an alias to an MCP server version.

    The reserved alias ``"latest"`` delegates to latest-version resolution logic.

    Args:
        name: Server name.
        alias: Alias name (e.g., ``"production"``).

    Returns:
        The :py:class:`MCPServerVersion <mlflow.entities.MCPServerVersion>`.
    """
    return MlflowClient().get_mcp_server_version_by_alias(name=name, alias=alias)


def get_latest_mcp_server_version(name: str) -> MCPServerVersion:
    """Get the latest version of an MCP server.

    Resolves using semantic-version ordering: highest semver among ACTIVE versions
    if one exists, otherwise highest semver among non-DELETED versions.

    Args:
        name: Server name.

    Returns:
        The latest :py:class:`MCPServerVersion <mlflow.entities.MCPServerVersion>`.
    """
    return MlflowClient().get_latest_mcp_server_version(name=name)


def search_mcp_server_versions(
    name: str,
    filter_string: str | None = None,
    max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
    order_by: list[str] | None = None,
    page_token: str | None = None,
) -> PagedList[MCPServerVersion]:
    """Search versions of a specific MCP server.

    Args:
        name: Server name.
        filter_string: SQL-like filter (e.g., ``"status = 'active'"``).
        max_results: Maximum number of results.
        order_by: List of columns to order by.
        page_token: Token for the next page.

    Returns:
        A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
        :py:class:`MCPServerVersion <mlflow.entities.MCPServerVersion>` objects.
    """
    return MlflowClient().search_mcp_server_versions(
        name=name,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )


def update_mcp_server_version(
    name: str,
    version: str,
    display_name: str | None = NOT_SET,
    status: Literal["draft", "active", "deprecated", "deleted"] | None = NOT_SET,
    tools: list[MCPTool] | None = NOT_SET,
) -> MCPServerVersion:
    """Update mutable fields of an MCP server version.

    Only fields that are explicitly passed are updated; omitted fields are left
    unchanged. Pass ``None`` to clear a field.

    Args:
        name: Server name.
        version: Version string.
        display_name: New display name. Pass ``None`` to clear.
        status: New status (``"draft"``, ``"active"``, ``"deprecated"``, ``"deleted"``).
            Transition rules are enforced.
        tools: New tool definitions. Pass ``None`` to clear.

    Returns:
        The updated :py:class:`MCPServerVersion <mlflow.entities.MCPServerVersion>`.
    """
    return MlflowClient().update_mcp_server_version(
        name=name,
        version=version,
        display_name=display_name,
        status=_parse_enum(status, MCPStatus, "status"),
        tools=tools,
    )


def delete_mcp_server_version(name: str, version: str) -> None:
    MlflowClient().delete_mcp_server_version(name=name, version=version)


# --- MCPAccessBinding CRUD ---


def create_mcp_access_binding(
    server_name: str,
    endpoint_url: str,
    transport_type: Literal["streamable-http", "sse"] = "streamable-http",
    server_version: str | None = None,
    server_alias: str | None = None,
) -> MCPAccessBinding:
    """Record an approved direct-access binding for an MCP server.

    Exactly one of ``server_version`` or ``server_alias`` must be provided.

    Args:
        server_name: Server name.
        endpoint_url: URL of the remote MCP endpoint.
        transport_type: Transport protocol — ``"streamable-http"`` (default) or ``"sse"``.
        server_version: Pin the binding to a specific version string.
        server_alias: Pin the binding to an alias.

    Returns:
        The created :py:class:`MCPAccessBinding <mlflow.entities.MCPAccessBinding>`.

    Example:

    .. code-block:: python

        import mlflow

        binding = mlflow.genai.create_mcp_access_binding(
            server_name="io.github.anthropic/brave-search",
            endpoint_url="https://mcp.acme.internal/brave-search",
            transport_type="streamable-http",
            server_alias="production",
        )
    """
    return MlflowClient().create_mcp_access_binding(
        server_name=server_name,
        endpoint_url=endpoint_url,
        transport_type=_parse_enum(transport_type, MCPRemoteTransportType, "transport_type"),
        server_version=server_version,
        server_alias=server_alias,
    )


def get_mcp_access_binding(server_name: str, binding_id: int) -> MCPAccessBinding:
    return MlflowClient().get_mcp_access_binding(server_name=server_name, binding_id=binding_id)


def search_mcp_access_bindings(
    server_name: str | None = None,
    server_version: str | None = None,
    server_alias: str | None = None,
    filter_string: str | None = None,
    max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
    order_by: list[str] | None = None,
    page_token: str | None = None,
) -> PagedList[MCPAccessBinding]:
    """Search access bindings across the workspace.

    Args:
        server_name: If provided, limit results to this server.
        server_version: If provided, limit results to bindings targeting this version.
        server_alias: If provided, limit results to bindings targeting this alias.
        filter_string: SQL-like filter.
        max_results: Maximum number of results.
        order_by: List of columns to order by.
        page_token: Token for the next page.

    Returns:
        A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
        :py:class:`MCPAccessBinding <mlflow.entities.MCPAccessBinding>` objects.
    """
    return MlflowClient().search_mcp_access_bindings(
        server_name=server_name,
        server_version=server_version,
        server_alias=server_alias,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )


def update_mcp_access_binding(
    server_name: str,
    binding_id: int,
    endpoint_url: str | None = NOT_SET,
    transport_type: Literal["streamable-http", "sse"] | None = NOT_SET,
    server_version: str | None = NOT_SET,
    server_alias: str | None = NOT_SET,
) -> MCPAccessBinding:
    """Update an existing access binding.

    Only fields that are explicitly passed are updated; omitted fields are left
    unchanged. Pass ``None`` to clear a field.

    Args:
        server_name: Server name.
        binding_id: Binding ID.
        endpoint_url: New endpoint URL. Pass ``None`` to clear.
        transport_type: New transport type. Pass ``None`` to clear.
        server_version: New version target. Pass ``None`` to clear.
        server_alias: New alias target. Pass ``None`` to clear.

    Returns:
        The updated :py:class:`MCPAccessBinding <mlflow.entities.MCPAccessBinding>`.
    """
    return MlflowClient().update_mcp_access_binding(
        server_name=server_name,
        binding_id=binding_id,
        endpoint_url=endpoint_url,
        transport_type=_parse_enum(transport_type, MCPRemoteTransportType, "transport_type"),
        server_version=server_version,
        server_alias=server_alias,
    )


def delete_mcp_access_binding(server_name: str, binding_id: int) -> None:
    MlflowClient().delete_mcp_access_binding(server_name=server_name, binding_id=binding_id)


# --- Tag operations ---


def set_mcp_server_tag(name: str, key: str, value: str) -> None:
    """Set a tag on an MCP server (upsert).

    Args:
        name: Server name.
        key: Tag key.
        value: Tag value.
    """
    MlflowClient().set_mcp_server_tag(name=name, key=key, value=value)


def delete_mcp_server_tag(name: str, key: str) -> None:
    MlflowClient().delete_mcp_server_tag(name=name, key=key)


def set_mcp_server_version_tag(name: str, version: str, key: str, value: str) -> None:
    """Set a tag on an MCP server version (upsert).

    Args:
        name: Server name.
        version: Version string.
        key: Tag key.
        value: Tag value.
    """
    MlflowClient().set_mcp_server_version_tag(name=name, version=version, key=key, value=value)


def delete_mcp_server_version_tag(name: str, version: str, key: str) -> None:
    MlflowClient().delete_mcp_server_version_tag(name=name, version=version, key=key)


# --- Alias operations ---


def set_mcp_server_alias(name: str, alias: str, version: str) -> None:
    """Set an alias on an MCP server pointing to a specific version (upsert).

    The alias name ``"latest"`` is reserved and cannot be set here.

    Args:
        name: Server name.
        alias: Alias name (e.g., ``"production"``).
        version: Target version string.
    """
    MlflowClient().set_mcp_server_alias(name=name, alias=alias, version=version)


def delete_mcp_server_alias(name: str, alias: str) -> None:
    MlflowClient().delete_mcp_server_alias(name=name, alias=alias)
