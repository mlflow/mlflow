from __future__ import annotations

import json
import urllib.request
from typing import TYPE_CHECKING, Any, Literal

from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPStatus, MCPTool
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking.mcp_server_registry.abstract_mixin import NOT_SET, MCPIcon
from mlflow.tracking.client import MlflowClient
from mlflow.utils.uri import get_uri_scheme

if TYPE_CHECKING:
    from mlflow.entities.mcp_access_binding import MCPAccessBinding
    from mlflow.entities.mcp_server import MCPServer
    from mlflow.entities.mcp_server_version import MCPServerVersion


def register_mcp_server(
    *,
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
    version = create_mcp_server_version(
        server_json=server_json,
        display_name=display_name,
        source=source,
        status=status,
        tools=tools,
    )

    if create_access_bindings_from_remotes:
        client = MlflowClient()
        for remote in server_json.get("remotes", []):
            url = remote.get("url")
            if not url:
                continue
            transport_str = remote.get("type", "streamable-http")
            try:
                transport = MCPRemoteTransportType(transport_str)
            except ValueError:
                transport = MCPRemoteTransportType.STREAMABLE_HTTP
            client.create_mcp_access_binding(
                server_name=version.name,
                endpoint_url=url,
                transport_type=transport,
                server_version=version.version,
            )

    return version


def register_mcp_server_from_url(
    *,
    url: str,
    display_name: str | None = None,
    source: str | None = None,
    status: Literal["draft", "active", "deprecated", "deleted"] = "draft",
    tools: list[MCPTool] | None = None,
    create_access_bindings_from_remotes: bool = False,
) -> MCPServerVersion:
    """Fetch a ``server.json`` from ``url`` and register the MCP server.

    Args:
        url: HTTPS URL pointing to a ``server.json`` document.
        display_name: Human-readable label for the version.
        source: Provenance URI; defaults to ``url`` when not provided.
        status: Initial status (default ``"draft"``).
        tools: Declared tool definitions for this version.
        create_access_bindings_from_remotes: When ``True``, create one direct-access
            binding per ``remotes[]`` entry in ``server_json``.

    Returns:
        The created :py:class:`MCPServerVersion <mlflow.entities.MCPServerVersion>`.
    """
    scheme = get_uri_scheme(url)
    if scheme not in ("http", "https"):
        raise MlflowException.invalid_parameter_value(
            f"URL must use http or https scheme, got: {scheme!r}"
        )

    with urllib.request.urlopen(url) as resp:
        server_json = json.loads(resp.read())

    return register_mcp_server(
        server_json=server_json,
        display_name=display_name,
        source=source or url,
        status=status,
        tools=tools,
        create_access_bindings_from_remotes=create_access_bindings_from_remotes,
    )


# --- MCPServer CRUD ---


def create_mcp_server(
    *,
    name: str,
    description: str | None = None,
    icons: list[MCPIcon] | None = None,
) -> MCPServer:
    """Create a new MCP server entry in the registry.

    Args:
        name: Unique server name in reverse-DNS format (e.g., ``"io.github.user/server"``).
        description: Human-readable description.
        icons: Sized icon variants following the upstream server.json icon schema.

    Returns:
        The created :py:class:`MCPServer <mlflow.entities.MCPServer>`.
    """
    return MlflowClient().create_mcp_server(name=name, description=description, icons=icons)


def get_mcp_server(*, name: str) -> MCPServer:
    """Get an MCP server by name.

    Args:
        name: Server name.

    Returns:
        The :py:class:`MCPServer <mlflow.entities.MCPServer>` with tags, aliases, and
        access bindings populated.
    """
    return MlflowClient().get_mcp_server(name=name)


def search_mcp_servers(
    *,
    filter_string: str | None = None,
    max_results: int = 100,
    order_by: list[str] | None = None,
    page_token: str | None = None,
) -> PagedList[MCPServer]:
    """Search MCP servers with optional filtering and pagination.

    Args:
        filter_string: SQL-like filter expression (e.g., ``"status = 'active'"``,
            ``"tags.team = 'platform'"``, ``"has_access_bindings = true"``).
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
    *,
    name: str,
    display_name: str | None = None,
    description: str | None = None,
    icons: list[MCPIcon] | None = None,
    latest_version: str | None = None,
) -> MCPServer:
    """Update mutable fields of an MCP server.

    Only provided (non-``None``) keyword arguments are updated.

    Args:
        name: Server name.
        display_name: New human-readable label.
        description: New description.
        icons: New icon variants.
        latest_version: Explicit version string to pin as ``@latest`` resolution.

    Returns:
        The updated :py:class:`MCPServer <mlflow.entities.MCPServer>`.
    """
    return MlflowClient().update_mcp_server(
        name=name,
        display_name=display_name if display_name is not None else NOT_SET,
        description=description if description is not None else NOT_SET,
        icons=icons if icons is not None else NOT_SET,
        latest_version=latest_version if latest_version is not None else NOT_SET,
    )


def delete_mcp_server(*, name: str) -> None:
    """Delete an MCP server and all its child entities.

    Args:
        name: Server name.
    """
    MlflowClient().delete_mcp_server(name=name)


# --- MCPServerVersion CRUD ---


def create_mcp_server_version(
    *,
    server_json: dict[str, Any],
    display_name: str | None = None,
    source: str | None = None,
    status: Literal["draft", "active", "deprecated", "deleted"] = "draft",
    tools: list[MCPTool] | None = None,
) -> MCPServerVersion:
    """Create a new version of an MCP server.

    The parent ``MCPServer`` is auto-created if it does not exist.

    Args:
        server_json: The canonical MCP ``server.json`` payload.
        display_name: Human-readable display name.
        source: Provenance URI.
        status: Initial status (default ``"draft"``).
        tools: Declared tools this server version provides.

    Returns:
        The created :py:class:`MCPServerVersion <mlflow.entities.MCPServerVersion>`.
    """
    return MlflowClient().create_mcp_server_version(
        server_json=server_json,
        display_name=display_name,
        source=source,
        status=MCPStatus(status) if status is not None else None,
        tools=tools,
    )


def get_mcp_server_version(*, name: str, version: str) -> MCPServerVersion:
    """Get a specific version of an MCP server.

    Args:
        name: Server name.
        version: Version string.

    Returns:
        The :py:class:`MCPServerVersion <mlflow.entities.MCPServerVersion>`.
    """
    return MlflowClient().get_mcp_server_version(name=name, version=version)


def get_mcp_server_version_by_alias(*, name: str, alias: str) -> MCPServerVersion:
    """Resolve an alias to an MCP server version.

    The reserved alias ``"latest"`` delegates to latest-version resolution logic.

    Args:
        name: Server name.
        alias: Alias name (e.g., ``"production"``).

    Returns:
        The :py:class:`MCPServerVersion <mlflow.entities.MCPServerVersion>`.
    """
    return MlflowClient().get_mcp_server_version_by_alias(name=name, alias=alias)


def get_latest_mcp_server_version(*, name: str) -> MCPServerVersion:
    """Get the latest version of an MCP server.

    Uses the server's pinned ``latest_version`` if set, otherwise returns the most recently
    created non-draft version.

    Args:
        name: Server name.

    Returns:
        The latest :py:class:`MCPServerVersion <mlflow.entities.MCPServerVersion>`.
    """
    return MlflowClient().get_latest_mcp_server_version(name=name)


def search_mcp_server_versions(
    *,
    name: str,
    filter_string: str | None = None,
    max_results: int = 100,
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
    *,
    name: str,
    version: str,
    display_name: str | None = None,
    status: Literal["draft", "active", "deprecated", "deleted"] | None = None,
    tools: list[MCPTool] | None = None,
) -> MCPServerVersion:
    """Update mutable fields of an MCP server version.

    Args:
        name: Server name.
        version: Version string.
        display_name: New display name.
        status: New status (``"draft"``, ``"active"``, ``"deprecated"``, ``"deleted"``).
            Transition rules are enforced.
        tools: New tool definitions.

    Returns:
        The updated :py:class:`MCPServerVersion <mlflow.entities.MCPServerVersion>`.
    """
    return MlflowClient().update_mcp_server_version(
        name=name,
        version=version,
        display_name=display_name if display_name is not None else NOT_SET,
        status=MCPStatus(status) if status is not None else NOT_SET,
        tools=tools if tools is not None else NOT_SET,
    )


def delete_mcp_server_version(*, name: str, version: str) -> None:
    """Delete a specific version of an MCP server.

    Args:
        name: Server name.
        version: Version string.
    """
    MlflowClient().delete_mcp_server_version(name=name, version=version)


# --- MCPAccessBinding CRUD ---


def create_mcp_access_binding(
    *,
    server_name: str,
    endpoint_url: str,
    transport_type: str = "streamable-http",
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
        transport_type=MCPRemoteTransportType(transport_type),
        server_version=server_version,
        server_alias=server_alias,
    )


def get_mcp_access_binding(*, server_name: str, binding_id: int) -> MCPAccessBinding:
    """Get a specific access binding.

    Args:
        server_name: Server name.
        binding_id: Binding ID.

    Returns:
        The :py:class:`MCPAccessBinding <mlflow.entities.MCPAccessBinding>`.
    """
    return MlflowClient().get_mcp_access_binding(server_name=server_name, binding_id=binding_id)


def search_mcp_access_bindings(
    *,
    server_name: str | None = None,
    server_version: str | None = None,
    server_alias: str | None = None,
    filter_string: str | None = None,
    max_results: int = 100,
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
    *,
    server_name: str,
    binding_id: int,
    endpoint_url: str | None = None,
    transport_type: str | None = None,
    server_version: str | None = None,
    server_alias: str | None = None,
) -> MCPAccessBinding:
    """Update an existing access binding.

    Args:
        server_name: Server name.
        binding_id: Binding ID.
        endpoint_url: New endpoint URL.
        transport_type: New transport type.
        server_version: New version target.
        server_alias: New alias target.

    Returns:
        The updated :py:class:`MCPAccessBinding <mlflow.entities.MCPAccessBinding>`.
    """
    return MlflowClient().update_mcp_access_binding(
        server_name=server_name,
        binding_id=binding_id,
        endpoint_url=endpoint_url if endpoint_url is not None else NOT_SET,
        transport_type=(
            MCPRemoteTransportType(transport_type) if transport_type is not None else NOT_SET
        ),
        server_version=server_version if server_version is not None else NOT_SET,
        server_alias=server_alias if server_alias is not None else NOT_SET,
    )


def delete_mcp_access_binding(*, server_name: str, binding_id: int) -> None:
    """Delete an access binding.

    Args:
        server_name: Server name.
        binding_id: Binding ID.
    """
    MlflowClient().delete_mcp_access_binding(server_name=server_name, binding_id=binding_id)


# --- Tag operations ---


def set_mcp_server_tag(*, name: str, key: str, value: str) -> None:
    """Set a tag on an MCP server (upsert).

    Args:
        name: Server name.
        key: Tag key.
        value: Tag value.
    """
    MlflowClient().set_mcp_server_tag(name=name, key=key, value=value)


def delete_mcp_server_tag(*, name: str, key: str) -> None:
    """Delete a tag from an MCP server.

    Args:
        name: Server name.
        key: Tag key.
    """
    MlflowClient().delete_mcp_server_tag(name=name, key=key)


def set_mcp_server_version_tag(*, name: str, version: str, key: str, value: str) -> None:
    """Set a tag on an MCP server version (upsert).

    Args:
        name: Server name.
        version: Version string.
        key: Tag key.
        value: Tag value.
    """
    MlflowClient().set_mcp_server_version_tag(name=name, version=version, key=key, value=value)


def delete_mcp_server_version_tag(*, name: str, version: str, key: str) -> None:
    """Delete a tag from an MCP server version.

    Args:
        name: Server name.
        version: Version string.
        key: Tag key.
    """
    MlflowClient().delete_mcp_server_version_tag(name=name, version=version, key=key)


# --- Alias operations ---


def set_mcp_server_alias(*, name: str, alias: str, version: str) -> None:
    """Set an alias on an MCP server pointing to a specific version (upsert).

    The alias name ``"latest"`` is reserved and cannot be set here.

    Args:
        name: Server name.
        alias: Alias name (e.g., ``"production"``).
        version: Target version string.
    """
    MlflowClient().set_mcp_server_alias(name=name, alias=alias, version=version)


def delete_mcp_server_alias(*, name: str, alias: str) -> None:
    """Delete an alias from an MCP server.

    Args:
        name: Server name.
        alias: Alias name.
    """
    MlflowClient().delete_mcp_server_alias(name=name, alias=alias)
