from dataclasses import dataclass

from mlflow.utils.annotations import experimental


@experimental(version="3.15.0")
@dataclass(frozen=True)
class MCPServerAlias:
    name: str
    alias: str
    version: str
