from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mlflow.entities.mcp_server import MCPRemoteTransportType
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.workspace_utils import resolve_entity_workspace_name

if TYPE_CHECKING:
    from mlflow.entities.mcp_server_version import MCPServerVersion


@experimental(version="3.15.0")
@dataclass
class MCPAccessEndpoint:
    id: str
    server_name: str
    url: str
    transport_type: MCPRemoteTransportType = MCPRemoteTransportType.STREAMABLE_HTTP
    server_version: str | None = None
    server_alias: str | None = None
    resolved_version: MCPServerVersion | None = field(default=None, repr=False)
    workspace: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPAccessEndpoint:
        if not isinstance(data, dict):
            raise MlflowException.invalid_parameter_value(
                "Failed to parse MCP access endpoint response: expected a dictionary"
            )

        try:
            from mlflow.entities.mcp_server_version import MCPServerVersion

            return cls(
                id=data["id"],
                server_name=data["server_name"],
                url=data["url"],
                transport_type=MCPRemoteTransportType(
                    data.get("transport_type", "streamable-http")
                ),
                workspace=data.get("workspace"),
                server_version=data.get("server_version"),
                server_alias=data.get("server_alias"),
                resolved_version=(
                    None
                    if data.get("resolved_version") is None
                    else MCPServerVersion.from_dict(data["resolved_version"])
                ),
                created_by=data.get("created_by"),
                last_updated_by=data.get("last_updated_by"),
                creation_timestamp=data.get("creation_timestamp"),
                last_updated_timestamp=data.get("last_updated_timestamp"),
            )
        except KeyError as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse MCP access endpoint response: missing required field {e}"
            ) from None
        except (ValueError, TypeError) as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse MCP access endpoint response: {e}"
            ) from None
