from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.issues_pb2 import Issue as ProtoIssue


class IssueStatus(str, Enum):
    """Enum for status of an :py:class:`mlflow.entities.Issue`."""

    PENDING = "pending"
    REJECTED = "rejected"
    RESOLVED = "resolved"

    def __str__(self):
        return self.value


class IssueSeverity(str, Enum):
    """Enum for severity level of an :py:class:`mlflow.entities.Issue`."""

    NOT_AN_ISSUE = "not_an_issue"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    def __str__(self):
        return self.value

    @cached_property
    def _rank(self) -> int:
        """Return the ordinal rank for severity comparison."""
        return list(IssueSeverity).index(self)

    def __lt__(self, other) -> bool:
        if isinstance(other, IssueSeverity):
            return self._rank < other._rank
        return NotImplemented

    def __le__(self, other) -> bool:
        if isinstance(other, IssueSeverity):
            return self._rank <= other._rank
        return NotImplemented

    def __gt__(self, other) -> bool:
        if isinstance(other, IssueSeverity):
            return self._rank > other._rank
        return NotImplemented

    def __ge__(self, other) -> bool:
        if isinstance(other, IssueSeverity):
            return self._rank >= other._rank
        return NotImplemented


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

    status: IssueStatus
    """Issue status."""

    created_timestamp: int
    """Creation timestamp in milliseconds."""

    last_updated_timestamp: int
    """Last update timestamp in milliseconds."""

    severity: IssueSeverity | None = None
    """Severity level indicator."""

    root_causes: list[str] | None = None
    """Analysis of the root causes of the issue."""

    source_run_id: str | None = None
    """MLflow run ID that discovered this issue."""

    categories: list[str] | None = None
    """Categories of this issue."""

    created_by: str | None = None
    """Identifier for who created this issue."""

    def to_dictionary(self) -> dict[str, Any]:
        """Convert Issue to dictionary representation."""
        return {
            "issue_id": self.issue_id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "severity": self.severity.value if self.severity else None,
            "root_causes": self.root_causes,
            "source_run_id": self.source_run_id,
            "categories": self.categories,
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
            status=IssueStatus(issue_dict["status"]),
            created_timestamp=issue_dict["created_timestamp"],
            last_updated_timestamp=issue_dict["last_updated_timestamp"],
            severity=(
                IssueSeverity(issue_dict.get("severity")) if issue_dict.get("severity") else None
            ),
            root_causes=issue_dict.get("root_causes"),
            source_run_id=issue_dict.get("source_run_id"),
            categories=issue_dict.get("categories"),
            created_by=issue_dict.get("created_by"),
        )

    def to_proto(self) -> ProtoIssue:
        """Convert Issue to protobuf representation."""
        proto_issue = ProtoIssue()
        proto_issue.issue_id = self.issue_id
        proto_issue.experiment_id = self.experiment_id
        proto_issue.name = self.name
        proto_issue.description = self.description
        proto_issue.status = self.status.value
        proto_issue.created_timestamp = self.created_timestamp
        proto_issue.last_updated_timestamp = self.last_updated_timestamp

        if self.severity:
            proto_issue.severity = self.severity.value
        if self.root_causes:
            proto_issue.root_causes.extend(self.root_causes)
        if self.source_run_id:
            proto_issue.source_run_id = self.source_run_id
        if self.categories:
            proto_issue.categories.extend(self.categories)
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
            status=IssueStatus(proto.status),
            created_timestamp=proto.created_timestamp,
            last_updated_timestamp=proto.last_updated_timestamp,
            severity=IssueSeverity(proto.severity) if proto.severity else None,
            root_causes=list(proto.root_causes) or None,
            source_run_id=proto.source_run_id or None,
            categories=list(proto.categories) or None,
            created_by=proto.created_by or None,
        )
