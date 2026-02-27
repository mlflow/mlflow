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

    frequency: float
    """Frequency score indicating how often this issue occurs."""

    status: str
    """Issue status."""

    created_timestamp: int
    """Creation timestamp in milliseconds."""

    last_updated_timestamp: int
    """Last update timestamp in milliseconds."""

    run_id: str | None = None
    """MLflow run ID that discovered this issue."""

    root_cause: str | None = None
    """Analysis of the root cause of the issue."""

    confidence: str | None = None
    """Confidence level indicator."""

    rationale_examples: list[str] | None = None
    """List of rationale strings providing examples of the issue."""

    example_trace_ids: list[str] | None = None
    """List of example trace IDs."""

    trace_ids: list[str] | None = None
    """List of trace IDs associated with this issue."""

    created_by: str | None = None
    """Identifier for who created this issue."""

    def to_dictionary(self) -> dict[str, Any]:
        """Convert Issue to dictionary representation."""
        return {
            "issue_id": self.issue_id,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "name": self.name,
            "description": self.description,
            "root_cause": self.root_cause,
            "status": self.status,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "rationale_examples": self.rationale_examples,
            "example_trace_ids": self.example_trace_ids,
            "trace_ids": self.trace_ids,
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
            run_id=issue_dict.get("run_id"),
            name=issue_dict["name"],
            description=issue_dict["description"],
            root_cause=issue_dict.get("root_cause"),
            status=issue_dict["status"],
            frequency=issue_dict["frequency"],
            confidence=issue_dict.get("confidence"),
            rationale_examples=issue_dict.get("rationale_examples"),
            example_trace_ids=issue_dict.get("example_trace_ids"),
            trace_ids=issue_dict.get("trace_ids"),
            created_timestamp=issue_dict["created_timestamp"],
            last_updated_timestamp=issue_dict["last_updated_timestamp"],
            created_by=issue_dict.get("created_by"),
        )

    def to_proto(self) -> ProtoIssue:
        """Convert Issue to protobuf representation."""

        proto_issue = ProtoIssue()
        proto_issue.issue_id = self.issue_id
        proto_issue.experiment_id = self.experiment_id
        proto_issue.name = self.name
        proto_issue.description = self.description
        proto_issue.frequency = self.frequency
        proto_issue.status = self.status
        proto_issue.created_timestamp = self.created_timestamp
        proto_issue.last_updated_timestamp = self.last_updated_timestamp

        if self.run_id:
            proto_issue.run_id = self.run_id
        if self.root_cause:
            proto_issue.root_cause = self.root_cause
        if self.confidence:
            proto_issue.confidence = self.confidence
        if self.rationale_examples:
            proto_issue.rationale_examples.extend(self.rationale_examples)
        if self.example_trace_ids:
            proto_issue.example_trace_ids.extend(self.example_trace_ids)
        if self.trace_ids:
            proto_issue.trace_ids.extend(self.trace_ids)
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
            frequency=proto.frequency,
            status=proto.status,
            created_timestamp=proto.created_timestamp,
            last_updated_timestamp=proto.last_updated_timestamp,
            run_id=proto.run_id or None,
            root_cause=proto.root_cause or None,
            confidence=proto.confidence or None,
            rationale_examples=list(proto.rationale_examples) or None,
            example_trace_ids=list(proto.example_trace_ids) or None,
            trace_ids=list(proto.trace_ids) or None,
            created_by=proto.created_by or None,
        )
