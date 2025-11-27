"""Workspace entity shared between server and stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlflow.protos.service_pb2 import Workspace as ProtoWorkspace


@dataclass(frozen=True, slots=True)
class Workspace:
    """Minimal metadata describing a workspace."""

    name: str
    description: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"name": self.name, "description": self.description}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Workspace":
        return cls(name=payload["name"], description=payload.get("description"))

    def to_proto(self) -> ProtoWorkspace:
        workspace = ProtoWorkspace()
        workspace.name = self.name
        if self.description is not None:
            workspace.description = self.description
        return workspace

    @classmethod
    def from_proto(cls, proto: ProtoWorkspace) -> "Workspace":
        description = proto.description if proto.HasField("description") else None
        return cls(name=proto.name, description=description)
