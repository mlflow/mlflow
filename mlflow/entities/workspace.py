"""Workspace entity shared between server and stores."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from mlflow.protos.service_pb2 import Workspace as ProtoWorkspace


class WorkspaceDeletionMode(str, Enum):
    """Controls what happens to resources when a workspace is deleted."""

    SET_DEFAULT = "SET_DEFAULT"
    """Reassign all resources in the workspace to the default workspace."""

    CASCADE = "CASCADE"
    """Delete all resources in the workspace."""

    RESTRICT = "RESTRICT"
    """Refuse to delete the workspace if it still contains resources."""


@dataclass(frozen=True, slots=True)
class Workspace:
    """Minimal metadata describing a workspace."""

    name: str
    description: str | None = None
    default_artifact_root: str | None = None
    trace_archival_location: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "description": self.description,
            "default_artifact_root": self.default_artifact_root,
            "trace_archival_location": self.trace_archival_location,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Workspace":
        return cls(
            name=payload["name"],
            description=payload.get("description"),
            default_artifact_root=payload.get("default_artifact_root"),
            trace_archival_location=payload.get("trace_archival_location"),
        )

    def to_proto(self) -> ProtoWorkspace:
        workspace = ProtoWorkspace()
        workspace.name = self.name
        if self.description is not None:
            workspace.description = self.description
        if self.default_artifact_root is not None:
            workspace.default_artifact_root = self.default_artifact_root
        if self.trace_archival_location is not None and hasattr(
            workspace, "trace_archival_location"
        ):
            workspace.trace_archival_location = self.trace_archival_location
        return workspace

    @classmethod
    def from_proto(cls, proto: ProtoWorkspace) -> "Workspace":
        description = proto.description if proto.HasField("description") else None
        default_artifact_root = (
            proto.default_artifact_root if proto.HasField("default_artifact_root") else None
        )
        trace_archival_location = None
        if hasattr(proto, "trace_archival_location") and proto.HasField("trace_archival_location"):
            trace_archival_location = proto.trace_archival_location
        return cls(
            name=proto.name,
            description=description,
            default_artifact_root=default_artifact_root,
            trace_archival_location=trace_archival_location,
        )
