from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from mlflow.exceptions import MlflowException
from mlflow.utils.workspace_utils import resolve_entity_workspace_name

if TYPE_CHECKING:
    from mlflow.entities.mcp_access_binding import MCPAccessBinding


class MCPStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DELETED = "deleted"

    def __str__(self):
        return self.value


VALID_STATUS_TRANSITIONS: dict[MCPStatus, set[MCPStatus]] = {
    MCPStatus.DRAFT: {MCPStatus.ACTIVE, MCPStatus.DELETED},
    MCPStatus.ACTIVE: {MCPStatus.DRAFT, MCPStatus.DEPRECATED},
    MCPStatus.DEPRECATED: {MCPStatus.ACTIVE, MCPStatus.DELETED},
}


class MCPRemoteTransportType(str, Enum):
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"

    def __str__(self):
        return self.value


_MCP_SERVER_NAME_NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$")
_MCP_SERVER_NAME_SLUG_RE = re.compile(r"^[a-zA-Z0-9._-]+$")
_MCP_SERVER_RESERVED_SLUGS = {"aliases", "bindings", "tags", "versions"}


def validate_mcp_server_name(name: str) -> None:
    if not name:
        raise MlflowException.invalid_parameter_value("MCP server name must not be empty")

    try:
        namespace, slug = name.split("/")
    except ValueError:
        raise MlflowException.invalid_parameter_value(
            "Invalid MCP server name. Expected '<reverse-dns namespace>/<server slug>' "
            "such as 'com.example/server-name'."
        ) from None

    if not namespace or not slug:
        raise MlflowException.invalid_parameter_value(
            "Invalid MCP server name. Expected '<reverse-dns namespace>/<server slug>' "
            "such as 'com.example/server-name'."
        )

    if slug in _MCP_SERVER_RESERVED_SLUGS:
        raise MlflowException.invalid_parameter_value(
            "Invalid MCP server name. Expected '<reverse-dns namespace>/<server slug>' "
            "such as 'com.example/server-name'."
        )

    labels = namespace.split(".")
    if len(labels) < 2 or any(not label for label in labels):
        raise MlflowException.invalid_parameter_value(
            "Invalid MCP server name. Expected '<reverse-dns namespace>/<server slug>' "
            "such as 'com.example/server-name'."
        )

    if any(_MCP_SERVER_NAME_NAMESPACE_RE.fullmatch(label) is None for label in labels):
        raise MlflowException.invalid_parameter_value(
            "Invalid MCP server name. Expected '<reverse-dns namespace>/<server slug>' "
            "such as 'com.example/server-name'."
        )

    if _MCP_SERVER_NAME_SLUG_RE.fullmatch(slug) is None:
        raise MlflowException.invalid_parameter_value(
            "Invalid MCP server name. Expected '<reverse-dns namespace>/<server slug>' "
            "such as 'com.example/server-name'."
        )


@dataclass(frozen=True)
class MCPTool:
    name: str
    title: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    annotations: dict[str, Any] | None = None
    icons: list[dict[str, Any]] | None = None
    execution: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in {
                "name": self.name,
                "title": self.title,
                "description": self.description,
                "inputSchema": self.input_schema,
                "outputSchema": self.output_schema,
                "annotations": self.annotations,
                "icons": self.icons,
                "execution": self.execution,
            }.items()
            if v is not None
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPTool:
        try:
            name = data["name"]
        except KeyError:
            raise MlflowException.invalid_parameter_value(
                "Missing required key 'name' in MCPTool dictionary"
            ) from None
        return cls(
            name=name,
            title=data.get("title"),
            description=data.get("description"),
            input_schema=data.get("inputSchema"),
            output_schema=data.get("outputSchema"),
            annotations=data.get("annotations"),
            icons=data.get("icons"),
            execution=data.get("execution"),
        )


@dataclass(frozen=True)
class MCPServerTag:
    key: str
    value: str


@dataclass
class MCPServer:
    name: str
    display_name: str | None = None
    description: str | None = None
    icons: list[dict[str, Any]] | None = None
    workspace: str | None = None
    status: MCPStatus | None = None
    tags: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    access_bindings: list[MCPAccessBinding] = field(default_factory=list)
    latest_version: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)
