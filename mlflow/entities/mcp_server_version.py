from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlflow.entities.mcp_server import MCPStatus, MCPTool
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


@experimental(version="3.15.0")
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPServerVersion:
        if not isinstance(data, dict):
            raise MlflowException.invalid_parameter_value(
                "Failed to parse MCP server version response: expected a dictionary"
            )

        try:
            tools = []
            if data.get("tools") is not None:
                tools = [MCPTool.from_dict(tool) for tool in data["tools"]]

            return cls(
                name=data["name"],
                version=data["version"],
                server_json=data["server_json"],
                display_name=data.get("display_name"),
                status=MCPStatus(data["status"]) if data.get("status") else MCPStatus.DRAFT,
                tools=tools,
                aliases=data.get("aliases") or [],
                tags=data.get("tags") or {},
                source=data.get("source"),
                workspace=data.get("workspace"),
                created_by=data.get("created_by"),
                last_updated_by=data.get("last_updated_by"),
                creation_timestamp=data.get("creation_timestamp"),
                last_updated_timestamp=data.get("last_updated_timestamp"),
            )
        except KeyError as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse MCP server version response: missing required field {e}"
            ) from None
        except (ValueError, TypeError, MlflowException) as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse MCP server version response: {e}"
            ) from None
