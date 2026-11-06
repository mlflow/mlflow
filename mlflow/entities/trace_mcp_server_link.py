from dataclasses import dataclass


@dataclass(frozen=True)
class MCPServerSpanObservation:
    span_id: str
    mcp_name: str
    mcp_version: str | None = None


@dataclass(frozen=True)
class TraceMCPServerLink:
    workspace: str
    name: str
    version: str | None
