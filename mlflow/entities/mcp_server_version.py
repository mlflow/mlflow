from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlflow.entities.mcp_server import MCPStatus, MCPTool
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


@dataclass
class MCPServerVersion:
    name: str
    version: str
    server_json: dict[str, Any]
    display_name: str | None = None
    status: MCPStatus = MCPStatus.DRAFT
    tools: list[MCPTool] | None = None
    aliases: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    source: str | None = None
    workspace: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)
