from __future__ import annotations

from typing import Any, TypedDict

from typing_extensions import NotRequired

from mlflow.entities.mcp_access_binding import MCPAccessBinding
from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPServer, MCPStatus, MCPTool
from mlflow.entities.mcp_server_version import MCPServerVersion
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT

NOT_SET = object()


class MCPIcon(TypedDict):
    """Icon following the upstream MCP server.json icon schema."""

    src: str
    sizes: NotRequired[list[str]]
    mimeType: NotRequired[str]
    theme: NotRequired[str]


class MCPServerRegistryMixin:
    """Mixin class providing MCP Server Registry interface for tracking stores.

    This mixin adds MCP server management to tracking stores, enabling
    registration, versioning, aliasing, tagging, and access binding
    management for MCP servers.

    All methods raise NotImplementedError rather than using @abstractmethod,
    following the GatewayStoreMixin pattern. This allows stores that don't
    support MCP servers (e.g., FileStore) to work without stubbing every method.
    """

    # --- MCPServer operations ---

    def create_mcp_server(
        self,
        name: str,
        description: str | None = None,
        icons: list[MCPIcon] | None = None,
    ) -> MCPServer:
        """Create a new MCP server entry.

        Args:
            name: Unique server name (e.g., "io.github.org/server").
            description: Human-readable description.
            icons: Sized icon variants (src, sizes, mimeType, theme).

        Returns:
            The created MCPServer entity.
        """
        raise NotImplementedError(self.__class__.__name__)

    def get_mcp_server(self, name: str) -> MCPServer:
        """Retrieve an MCP server by name.

        Args:
            name: Server name.

        Returns:
            The MCPServer entity with tags, aliases, and bindings populated.
        """
        raise NotImplementedError(self.__class__.__name__)

    def search_mcp_servers(
        self,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPServer]:
        """Search MCP servers with optional filtering and pagination.

        Args:
            filter_string: SQL-like filter (e.g., "status = 'active'").
            max_results: Maximum number of results to return.
            order_by: List of columns to order by.
            page_token: Token for pagination.

        Returns:
            A PagedList of MCPServer entities.
        """
        raise NotImplementedError(self.__class__.__name__)

    def update_mcp_server(
        self,
        name: str,
        description: str | None = NOT_SET,
        display_name: str | None = NOT_SET,
        icons: list[MCPIcon] | None = NOT_SET,
        latest_version: str | None = NOT_SET,
    ) -> MCPServer:
        """Update an existing MCP server's metadata.

        Args:
            name: Server name.
            description: New description (if not None).
            display_name: New display name (if not None).
            icons: New icon variants (if not None).
            latest_version: Pin the latest version pointer (if not None).

        Returns:
            The updated MCPServer entity.
        """
        raise NotImplementedError(self.__class__.__name__)

    def delete_mcp_server(self, name: str) -> None:
        """Delete an MCP server and all its child entities.

        Args:
            name: Server name.
        """
        raise NotImplementedError(self.__class__.__name__)

    # --- MCPServerVersion operations ---

    def create_mcp_server_version(
        self,
        server_json: dict[str, Any],
        display_name: str | None = None,
        source: str | None = None,
        status: MCPStatus | None = None,
        tools: list[MCPTool] | None = None,
    ) -> MCPServerVersion:
        """Create a new version of an MCP server.

        The parent MCPServer is auto-created from server_json["name"] if it
        does not already exist.

        Args:
            server_json: The server.json payload (must contain "name" and "version").
            display_name: Human-readable display name.
            source: Origin URL or identifier for this version.
            status: Initial status (defaults to DRAFT).
            tools: List of MCPTool definitions.

        Returns:
            The created MCPServerVersion entity.
        """
        raise NotImplementedError(self.__class__.__name__)

    def get_mcp_server_version(self, name: str, version: str) -> MCPServerVersion:
        """Retrieve a specific version of an MCP server.

        Args:
            name: Server name.
            version: Version string.

        Returns:
            The MCPServerVersion entity.
        """
        raise NotImplementedError(self.__class__.__name__)

    def get_mcp_server_version_by_alias(self, name: str, alias: str) -> MCPServerVersion:
        """Retrieve a version by its alias. "latest" delegates to get_latest.

        Args:
            name: Server name.
            alias: Alias name.

        Returns:
            The MCPServerVersion entity pointed to by the alias.
        """
        raise NotImplementedError(self.__class__.__name__)

    def get_latest_mcp_server_version(self, name: str) -> MCPServerVersion:
        """Retrieve the latest version of an MCP server.

        Uses the pinned latest_version if set, otherwise falls back to the
        most recently created non-draft version.

        Args:
            name: Server name.

        Returns:
            The latest MCPServerVersion entity.
        """
        raise NotImplementedError(self.__class__.__name__)

    def search_mcp_server_versions(
        self,
        name: str,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPServerVersion]:
        """Search versions of a specific MCP server.

        Args:
            name: Server name.
            filter_string: SQL-like filter (e.g., "status = 'active'").
            max_results: Maximum number of results to return.
            order_by: List of columns to order by.
            page_token: Token for pagination.

        Returns:
            A PagedList of MCPServerVersion entities.
        """
        raise NotImplementedError(self.__class__.__name__)

    def update_mcp_server_version(
        self,
        name: str,
        version: str,
        display_name: str | None = NOT_SET,
        status: MCPStatus | None = NOT_SET,
        tools: list[MCPTool] | None = NOT_SET,
    ) -> MCPServerVersion:
        """Update a version's metadata or status.

        Args:
            name: Server name.
            version: Version string.
            display_name: New display name (if not None).
            status: New status (if not None); validated against transition rules.
            tools: New tool definitions (if not None).

        Returns:
            The updated MCPServerVersion entity.
        """
        raise NotImplementedError(self.__class__.__name__)

    def delete_mcp_server_version(self, name: str, version: str) -> None:
        """Delete a specific version of an MCP server.

        Args:
            name: Server name.
            version: Version string.
        """
        raise NotImplementedError(self.__class__.__name__)

    # --- MCPAccessBinding operations ---

    def create_mcp_access_binding(
        self,
        server_name: str,
        endpoint_url: str,
        transport_type: MCPRemoteTransportType = MCPRemoteTransportType.STREAMABLE_HTTP,
        server_version: str | None = None,
        server_alias: str | None = None,
    ) -> MCPAccessBinding:
        """Create a direct-access binding for an MCP server.

        Exactly one of server_version or server_alias must be set.

        Args:
            server_name: Server name.
            endpoint_url: URL of the remote MCP endpoint.
            transport_type: Transport protocol.
            server_version: Pin to a specific version.
            server_alias: Pin to an alias.

        Returns:
            The created MCPAccessBinding entity.
        """
        raise NotImplementedError(self.__class__.__name__)

    def get_mcp_access_binding(self, server_name: str, binding_id: int) -> MCPAccessBinding:
        """Retrieve a specific access binding.

        Args:
            server_name: Server name.
            binding_id: Binding ID.

        Returns:
            The MCPAccessBinding entity.
        """
        raise NotImplementedError(self.__class__.__name__)

    def search_mcp_access_bindings(
        self,
        server_name: str | None = None,
        server_version: str | None = None,
        server_alias: str | None = None,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPAccessBinding]:
        """Search access bindings, optionally scoped to a server.

        Args:
            server_name: If set, only return bindings for this server.
            server_version: If set, only return bindings targeting this version.
            server_alias: If set, only return bindings targeting this alias.
            filter_string: SQL-like filter.
            max_results: Maximum number of results to return.
            order_by: List of columns to order by.
            page_token: Token for pagination.

        Returns:
            A PagedList of MCPAccessBinding entities.
        """
        raise NotImplementedError(self.__class__.__name__)

    def update_mcp_access_binding(
        self,
        server_name: str,
        binding_id: int,
        server_version: str | None = NOT_SET,
        server_alias: str | None = NOT_SET,
        endpoint_url: str | None = NOT_SET,
        transport_type: MCPRemoteTransportType | None = NOT_SET,
    ) -> MCPAccessBinding:
        """Update an existing access binding.

        Args:
            server_name: Server name.
            binding_id: Binding ID.
            server_version: New version target (if not None).
            server_alias: New alias target (if not None).
            endpoint_url: New endpoint URL (if not None).
            transport_type: New transport type (if not None).

        Returns:
            The updated MCPAccessBinding entity.
        """
        raise NotImplementedError(self.__class__.__name__)

    def delete_mcp_access_binding(self, server_name: str, binding_id: int) -> None:
        """Delete an access binding.

        Args:
            server_name: Server name.
            binding_id: Binding ID.
        """
        raise NotImplementedError(self.__class__.__name__)

    # --- Tag operations ---

    def set_mcp_server_tag(self, name: str, key: str, value: str) -> None:
        """Set a tag on an MCP server (upsert)."""
        raise NotImplementedError(self.__class__.__name__)

    def delete_mcp_server_tag(self, name: str, key: str) -> None:
        """Delete a tag from an MCP server."""
        raise NotImplementedError(self.__class__.__name__)

    def set_mcp_server_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        """Set a tag on an MCP server version (upsert)."""
        raise NotImplementedError(self.__class__.__name__)

    def delete_mcp_server_version_tag(self, name: str, version: str, key: str) -> None:
        """Delete a tag from an MCP server version."""
        raise NotImplementedError(self.__class__.__name__)

    # --- Alias operations ---

    def set_mcp_server_alias(self, name: str, alias: str, version: str) -> None:
        """Set an alias on an MCP server pointing to a version (upsert).

        The alias name ``"latest"`` is reserved for automatic resolution and
        must not be used here.  Implementations should raise
        ``MlflowException`` with ``INVALID_PARAMETER_VALUE`` if the caller
        passes ``alias="latest"``.
        """
        raise NotImplementedError(self.__class__.__name__)

    def delete_mcp_server_alias(self, name: str, alias: str) -> None:
        """Delete an alias from an MCP server."""
        raise NotImplementedError(self.__class__.__name__)

    # --- Trace linking ---

    def link_mcp_server_versions_to_trace(
        self,
        trace_id: str,
        mcp_servers: list[MCPServerVersion],
    ) -> None:
        """Link MCP server versions to a trace via entity associations."""
        raise NotImplementedError(self.__class__.__name__)

    def get_mcp_server_versions_for_trace(
        self,
        trace_id: str,
    ) -> list[MCPServerVersion]:
        """Retrieve MCP server versions linked to a trace."""
        raise NotImplementedError(self.__class__.__name__)
