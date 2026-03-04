from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.issues_pb2 import Issue as ProtoIssue


@dataclass
class Issue(_MlflowObject):
    """
    An Issue represents a quality or operational problem discovered in traces.
    """

    issue_id: str
    """Unique identifier for the issue."""

    experiment_id: str
    """Experiment ID."""

    name: str
    """Short descriptive name for the issue."""

    description: str
    """Detailed description of the issue."""

    status: str
    """Issue status."""

    created_timestamp: int
    """Creation timestamp in milliseconds."""

    last_updated_timestamp: int
    """Last update timestamp in milliseconds."""

    confidence: str | None = None
    """Confidence level indicator."""

    root_causes: list[str] | None = None
    """Analysis of the root causes of the issue."""

    source_run_id: str | None = None
    """MLflow run ID that discovered this issue."""

    created_by: str | None = None
    """Identifier for who created this issue."""

    def to_dictionary(self) -> dict[str, Any]:
        """Convert Issue to dictionary representation."""
        return {
            "issue_id": self.issue_id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "confidence": self.confidence,
            "root_causes": self.root_causes,
            "source_run_id": self.source_run_id,
            "created_timestamp": self.created_timestamp,
            "last_updated_timestamp": self.last_updated_timestamp,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dictionary(cls, issue_dict: dict[str, Any]) -> Issue:
        """Create Issue from dictionary representation."""
        return cls(
            issue_id=issue_dict["issue_id"],
            experiment_id=issue_dict["experiment_id"],
            name=issue_dict["name"],
            description=issue_dict["description"],
            status=issue_dict["status"],
            created_timestamp=issue_dict["created_timestamp"],
            last_updated_timestamp=issue_dict["last_updated_timestamp"],
            confidence=issue_dict.get("confidence"),
            root_causes=issue_dict.get("root_causes"),
            source_run_id=issue_dict.get("source_run_id"),
            created_by=issue_dict.get("created_by"),
        )

    def to_proto(self) -> ProtoIssue:
        """Convert Issue to protobuf representation."""
        proto_issue = ProtoIssue()
        proto_issue.issue_id = self.issue_id
        proto_issue.experiment_id = self.experiment_id
        proto_issue.name = self.name
        proto_issue.description = self.description
        proto_issue.status = self.status
        proto_issue.created_timestamp = self.created_timestamp
        proto_issue.last_updated_timestamp = self.last_updated_timestamp

        if self.confidence:
            proto_issue.confidence = self.confidence
        if self.root_causes:
            proto_issue.root_causes.extend(self.root_causes)
        if self.source_run_id:
            proto_issue.source_run_id = self.source_run_id
        if self.created_by:
            proto_issue.created_by = self.created_by

        return proto_issue

    @classmethod
    def from_proto(cls, proto: ProtoIssue) -> Issue:
        """Create Issue from protobuf representation."""
        return cls(
            issue_id=proto.issue_id,
            experiment_id=proto.experiment_id,
            name=proto.name,
            description=proto.description,
            status=proto.status,
            created_timestamp=proto.created_timestamp,
            last_updated_timestamp=proto.last_updated_timestamp,
            confidence=proto.confidence or None,
            root_causes=list(proto.root_causes) or None,
            source_run_id=proto.source_run_id or None,
            created_by=proto.created_by or None,
        )
