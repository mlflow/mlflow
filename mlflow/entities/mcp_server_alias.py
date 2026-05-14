from dataclasses import dataclass


@dataclass(frozen=True)
class MCPServerAlias:
    name: str
    alias: str
    version: str
