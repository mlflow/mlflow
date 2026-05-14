from __future__ import annotations

from dataclasses import dataclass

from mlflow.entities.mcp_server import MCPRemoteTransportType
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


@dataclass
class MCPAccessBinding:
    binding_id: int
    server_name: str
    endpoint_url: str
    transport_type: MCPRemoteTransportType = MCPRemoteTransportType.STREAMABLE_HTTP
    server_version: str | None = None
    server_alias: str | None = None
    workspace: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)
