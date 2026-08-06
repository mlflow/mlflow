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
class TraceArchivalConfig:
    """Python-facing configuration for workspace trace archival.

    Use ``location`` for the archival storage URI/root and ``retention`` for the retention
    duration formatted as ``<int><unit>`` such as ``30d`` or ``12h``.

    ``None`` leaves a field unset. For update-style APIs, use an empty string to clear an
    existing value while leaving a field as ``None`` keeps the current value unchanged.
    """

    location: str | None = None
    retention: str | None = None


@dataclass(frozen=True, slots=True)
class Workspace:
    """Minimal metadata describing a workspace."""

    name: str
    description: str | None = None
    default_artifact_root: str | None = None
    trace_archival_location: str | None = None
    trace_archival_retention: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "default_artifact_root": self.default_artifact_root,
        }
        trace_archival_config = {}
        if self.trace_archival_location is not None:
            trace_archival_config["location"] = self.trace_archival_location
        if self.trace_archival_retention is not None:
            trace_archival_config["retention"] = self.trace_archival_retention
        if trace_archival_config:
            payload["trace_archival_config"] = trace_archival_config
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Workspace":
        trace_archival_config = payload.get("trace_archival_config") or {}
        return cls(
            name=payload["name"],
            description=payload.get("description"),
            default_artifact_root=payload.get("default_artifact_root"),
            trace_archival_location=trace_archival_config.get("location"),
            trace_archival_retention=trace_archival_config.get("retention"),
        )

    def to_proto(self) -> ProtoWorkspace:
        workspace = ProtoWorkspace()
        workspace.name = self.name
        if self.description is not None:
            workspace.description = self.description
        if self.default_artifact_root is not None:
            workspace.default_artifact_root = self.default_artifact_root
        if self.trace_archival_location is not None:
            workspace.trace_archival_config.location = self.trace_archival_location
        if self.trace_archival_retention is not None:
            workspace.trace_archival_config.retention = self.trace_archival_retention
        return workspace

    @classmethod
    def from_proto(cls, proto: ProtoWorkspace) -> "Workspace":
        description = proto.description if proto.HasField("description") else None
        default_artifact_root = (
            proto.default_artifact_root if proto.HasField("default_artifact_root") else None
        )
        trace_archival_location = None
        trace_archival_retention = None
        if proto.HasField("trace_archival_config"):
            if proto.trace_archival_config.HasField("location"):
                trace_archival_location = proto.trace_archival_config.location
            if proto.trace_archival_config.HasField("retention"):
                trace_archival_retention = proto.trace_archival_config.retention
        return cls(
            name=proto.name,
            description=description,
            default_artifact_root=default_artifact_root,
            trace_archival_location=trace_archival_location,
            trace_archival_retention=trace_archival_retention,
        )
