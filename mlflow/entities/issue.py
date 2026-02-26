from __future__ import annotations

from dataclasses import dataclass

from mlflow.entities._mlflow_object import _MlflowObject


@dataclass
class Issue(_MlflowObject):
    """
    An Issue represents a quality or operational problem discovered in traces.
    """

    issue_id: str
    """Issue ID: Unique identifier for the issue (format: "iss-<uuid>")."""

    name: str
    """Issue name/title: Short descriptive name for the issue."""

    description: str
    """Detailed description of the issue."""

    frequency: float
    """Frequency score (0.0 to 1.0) indicating how often this issue occurs."""

    status: str
    """Issue status: One of "draft", "accepted", or "rejected"."""

    created_timestamp: int
    """Creation timestamp in milliseconds."""

    last_updated_timestamp: int
    """Last update timestamp in milliseconds."""

    run_id: str | None = None
    """Optional MLflow run ID that discovered this issue."""

    root_cause: str | None = None
    """Optional analysis of the root cause of the issue."""

    confidence: str | None = None
    """Optional confidence level indicator."""

    rationale_examples: list[str] | None = None
    """Optional list of rationale strings providing examples of the issue."""

    example_trace_ids: list[str] | None = None
    """Optional list of example trace IDs that exemplify this issue (small subset for UI display)."""

    created_by: str | None = None
    """Optional identifier for who/what created this issue."""

    def to_dictionary(self) -> dict:
        """Convert Issue to dictionary representation."""
        return {
            "issue_id": self.issue_id,
            "run_id": self.run_id,
            "name": self.name,
            "description": self.description,
            "root_cause": self.root_cause,
            "status": self.status,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "rationale_examples": self.rationale_examples,
            "example_trace_ids": self.example_trace_ids,
            "created_timestamp": self.created_timestamp,
            "last_updated_timestamp": self.last_updated_timestamp,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dictionary(cls, issue_dict: dict) -> Issue:
        """Create Issue from dictionary representation."""
        return cls(
            issue_id=issue_dict["issue_id"],
            run_id=issue_dict.get("run_id"),
            name=issue_dict["name"],
            description=issue_dict["description"],
            root_cause=issue_dict.get("root_cause"),
            status=issue_dict["status"],
            frequency=issue_dict["frequency"],
            confidence=issue_dict.get("confidence"),
            rationale_examples=issue_dict.get("rationale_examples"),
            example_trace_ids=issue_dict.get("example_trace_ids"),
            created_timestamp=issue_dict["created_timestamp"],
            last_updated_timestamp=issue_dict["last_updated_timestamp"],
            created_by=issue_dict.get("created_by"),
        )
