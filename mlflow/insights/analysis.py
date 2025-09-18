"""
Analysis, Hypothesis, and Issue models for MLflow Insights.

This module provides data models for tracking analyses, hypotheses, and issues
discovered during ML model investigation and experimentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from mlflow.exceptions import MlflowException
from mlflow.insights.constants import AnalysisStatus, HypothesisStatus, IssueSeverity, IssueStatus
from mlflow.insights.models import EvidenceEntry
from mlflow.insights.utils import extract_trace_ids, normalize_evidence


class TimestampedModel(BaseModel, ABC):
    """Base class for models with timestamp tracking."""

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp in UTC"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp in UTC"
    )

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp to current UTC time."""
        self.updated_at = datetime.utcnow()


class ExtensibleModel(BaseModel, ABC):
    """Base class for models with extensible metadata."""

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible dictionary for custom fields and future extensions"
    )

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure metadata is a dictionary."""
        if not isinstance(v, dict):
            raise MlflowException.invalid_parameter_value(
                f"metadata must be a dictionary, got {type(v).__name__}"
            )
        return v


class EvidencedModel(BaseModel, ABC):
    """Base class for models that contain evidence entries."""

    evidence: list[EvidenceEntry] = Field(
        default_factory=list,
        description="List of evidence entries linking to MLflow traces"
    )

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence_field(cls, v: Any) -> list[EvidenceEntry]:
        """Normalize evidence to list of EvidenceEntry objects."""
        # Determine if this is for an issue (no supports field)
        for_issue = cls.__name__ == "Issue"
        return normalize_evidence(v, for_issue=for_issue)

    @property
    def trace_count(self) -> int:
        """Get the number of unique traces associated with this model."""
        return len(extract_trace_ids(self.evidence))

    @property
    def evidence_count(self) -> int:
        """Get the total number of evidence entries."""
        return len(self.evidence)

    def get_trace_ids(self) -> list[str]:
        """Get list of unique trace IDs from evidence."""
        return extract_trace_ids(self.evidence)


class Analysis(TimestampedModel, ExtensibleModel):
    """
    High-level investigation or analysis session.

    An Analysis represents a top-level investigation with goals, guidance,
    and current status. It's stored as 'analysis.yaml' in the insights
    artifact directory of an MLflow run.
    """

    name: str = Field(
        description="Human-readable name for the analysis"
    )
    description: str = Field(
        description="Detailed description of investigation goals, approach, and guidance"
    )
    status: AnalysisStatus = Field(
        default=AnalysisStatus.ACTIVE,
        description="Current status of the analysis"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is not empty."""
        if not v or not v.strip():
            raise MlflowException.invalid_parameter_value("Analysis name cannot be empty")
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate description is not empty."""
        if not v or not v.strip():
            raise MlflowException.invalid_parameter_value("Analysis description cannot be empty")
        return v.strip()

    def complete(self) -> None:
        """Mark the analysis as completed."""
        self.status = AnalysisStatus.COMPLETED
        self.update_timestamp()

    def archive(self) -> None:
        """Archive the analysis."""
        self.status = AnalysisStatus.ARCHIVED
        self.update_timestamp()

    def reactivate(self) -> None:
        """Reactivate an archived or completed analysis."""
        self.status = AnalysisStatus.ACTIVE
        self.update_timestamp()


class Hypothesis(TimestampedModel, ExtensibleModel, EvidencedModel):
    """
    Testable statement or theory with supporting/refuting evidence.

    A Hypothesis represents a specific testable claim being investigated,
    along with its testing plan and collected evidence. It's stored as
    'hypothesis_<id>.yaml' in the insights artifact directory.
    """

    hypothesis_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier (UUID) for this hypothesis"
    )
    statement: str = Field(
        description="The hypothesis statement being tested"
    )
    testing_plan: str = Field(
        description="Detailed plan for testing including validation/refutation criteria"
    )
    status: HypothesisStatus = Field(
        default=HypothesisStatus.TESTING,
        description="Current testing status"
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Quantitative metrics related to hypothesis testing"
    )

    @field_validator("statement")
    @classmethod
    def validate_statement(cls, v: str) -> str:
        """Validate statement is not empty."""
        if not v or not v.strip():
            raise MlflowException.invalid_parameter_value("Hypothesis statement cannot be empty")
        return v.strip()

    @field_validator("testing_plan")
    @classmethod
    def validate_testing_plan(cls, v: str) -> str:
        """Validate testing plan is not empty."""
        if not v or not v.strip():
            raise MlflowException.invalid_parameter_value("Hypothesis testing plan cannot be empty")
        return v.strip()

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure metrics is a dictionary."""
        if not isinstance(v, dict):
            raise MlflowException.invalid_parameter_value(
                f"metrics must be a dictionary, got {type(v).__name__}"
            )
        return v

    @property
    def supports_count(self) -> int:
        """Count of evidence entries that support the hypothesis."""
        return sum(1 for e in self.evidence if e.supports is True)

    @property
    def refutes_count(self) -> int:
        """Count of evidence entries that refute the hypothesis."""
        return sum(1 for e in self.evidence if e.supports is False)

    def add_evidence(self, trace_id: str, rationale: str, supports: bool = True) -> None:
        """
        Add supporting or refuting evidence for this hypothesis.

        Args:
            trace_id: MLflow trace ID containing relevant evidence
            rationale: Explanation of why this trace is relevant
            supports: Whether the evidence supports (True) or refutes (False) the hypothesis
        """
        entry = EvidenceEntry.for_hypothesis(trace_id, rationale, supports)
        self.evidence.append(entry)
        self.update_timestamp()

    def validate_hypothesis(self) -> None:
        """Mark the hypothesis as validated based on evidence."""
        self.status = HypothesisStatus.VALIDATED
        self.update_timestamp()

    def reject_hypothesis(self) -> None:
        """Mark the hypothesis as rejected based on evidence."""
        self.status = HypothesisStatus.REJECTED
        self.update_timestamp()

    def reopen_for_testing(self) -> None:
        """Reopen a validated/rejected hypothesis for additional testing."""
        self.status = HypothesisStatus.TESTING
        self.update_timestamp()

    def add_metric(self, key: str, value: Any) -> None:
        """Add or update a metric value."""
        self.metrics[key] = value
        self.update_timestamp()


class Issue(TimestampedModel, ExtensibleModel, EvidencedModel):
    """
    Validated problem discovered through investigation.

    An Issue represents a confirmed problem or concern discovered during
    analysis, with evidence and potential resolution. Issues are stored as
    'issue_<id>.yaml' in the parent/container run artifacts.
    """

    issue_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier (UUID) for this issue"
    )
    source_run_id: str = Field(
        description="MLflow run ID where this issue was discovered"
    )
    hypothesis_id: str | None = Field(
        default=None,
        description="Optional source hypothesis ID if issue was validated from a hypothesis"
    )
    title: str = Field(
        description="Brief, descriptive title for the issue"
    )
    description: str = Field(
        description="Detailed description of the problem and its impact"
    )
    severity: IssueSeverity = Field(
        description="Severity level of the issue"
    )
    status: IssueStatus = Field(
        default=IssueStatus.OPEN,
        description="Current issue status"
    )
    assessments: list[str] = Field(
        default_factory=list,
        description="List of assessment names/IDs related to this issue"
    )
    resolution: str | None = Field(
        default=None,
        description="Description of how the issue was resolved (when applicable)"
    )

    @field_validator("source_run_id")
    @classmethod
    def validate_source_run_id(cls, v: str) -> str:
        """Validate source_run_id is not empty."""
        if not v or not v.strip():
            raise MlflowException.invalid_parameter_value("Issue source_run_id cannot be empty")
        return v.strip()

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate title is not empty."""
        if not v or not v.strip():
            raise MlflowException.invalid_parameter_value("Issue title cannot be empty")
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate description is not empty."""
        if not v or not v.strip():
            raise MlflowException.invalid_parameter_value("Issue description cannot be empty")
        return v.strip()

    @field_validator("assessments")
    @classmethod
    def validate_assessments(cls, v: list[str]) -> list[str]:
        """Validate assessments list contains non-empty strings."""
        if not isinstance(v, list):
            raise MlflowException.invalid_parameter_value(
                f"assessments must be a list, got {type(v).__name__}"
            )
        cleaned = []
        for item in v:
            if not isinstance(item, str):
                raise MlflowException.invalid_parameter_value(
                    f"assessment items must be strings, got {type(item).__name__}"
                )
            stripped = item.strip()
            if stripped:
                cleaned.append(stripped)
        return cleaned

    def add_evidence(self, trace_id: str, rationale: str) -> None:
        """
        Add evidence demonstrating this issue.

        Args:
            trace_id: MLflow trace ID showing the issue
            rationale: Explanation of how this trace demonstrates the issue
        """
        entry = EvidenceEntry.for_issue(trace_id, rationale)
        self.evidence.append(entry)
        self.update_timestamp()

    def add_assessment(self, assessment_name: str) -> None:
        """
        Link an assessment to this issue.

        Args:
            assessment_name: Name/ID of the related assessment
        """
        if not assessment_name or not assessment_name.strip():
            raise MlflowException.invalid_parameter_value("Assessment name cannot be empty")

        cleaned = assessment_name.strip()
        if cleaned not in self.assessments:
            self.assessments.append(cleaned)
            self.update_timestamp()

    def start_progress(self) -> None:
        """Mark the issue as being worked on."""
        if self.status not in (IssueStatus.RESOLVED, IssueStatus.REJECTED):
            self.status = IssueStatus.IN_PROGRESS
            self.update_timestamp()

    def resolve(self, resolution: str) -> None:
        """
        Mark the issue as resolved with a resolution description.

        Args:
            resolution: Description of how the issue was resolved
        """
        if not resolution or not resolution.strip():
            raise MlflowException.invalid_parameter_value("Resolution description cannot be empty")

        self.status = IssueStatus.RESOLVED
        self.resolution = resolution.strip()
        self.update_timestamp()

    def reject(self, reason: str | None = None) -> None:
        """
        Reject the issue as invalid or not actionable.

        Args:
            reason: Optional explanation for rejection
        """
        self.status = IssueStatus.REJECTED
        if reason:
            self.resolution = f"Rejected: {reason.strip()}"
        self.update_timestamp()

    def reopen(self) -> None:
        """Reopen a resolved or rejected issue."""
        self.status = IssueStatus.OPEN
        self.resolution = None
        self.update_timestamp()


# Summary models for efficient list operations
class BaseSummary(BaseModel, ABC):
    """Base class for summary models."""

    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    @abstractmethod
    def get_id(self) -> str:
        """Get the unique identifier for this summary."""
        pass


class AnalysisSummary(BaseSummary):
    """Lightweight summary of an Analysis for list operations."""

    run_id: str = Field(description="MLflow run ID containing this analysis")
    name: str = Field(description="Analysis name")
    description: str = Field(description="Analysis description")
    status: AnalysisStatus = Field(description="Current status")
    hypothesis_count: int = Field(default=0, description="Number of hypotheses")
    validated_count: int = Field(default=0, description="Number of validated hypotheses")
    rejected_count: int = Field(default=0, description="Number of rejected hypotheses")

    def get_id(self) -> str:
        """Get the unique identifier (run_id) for this analysis."""
        return self.run_id

    @classmethod
    def from_analysis(
        cls,
        run_id: str,
        analysis: Analysis,
        hypotheses: list[Hypothesis] | None = None
    ) -> AnalysisSummary:
        """
        Create a summary from an Analysis instance.

        Args:
            run_id: MLflow run ID containing the analysis
            analysis: The Analysis instance to summarize
            hypotheses: Optional list of hypotheses to calculate counts
        """
        hypothesis_count = 0
        validated_count = 0
        rejected_count = 0

        if hypotheses:
            hypothesis_count = len(hypotheses)
            validated_count = sum(1 for h in hypotheses if h.status == HypothesisStatus.VALIDATED)
            rejected_count = sum(1 for h in hypotheses if h.status == HypothesisStatus.REJECTED)

        return cls(
            run_id=run_id,
            name=analysis.name,
            description=analysis.description,
            status=analysis.status,
            created_at=analysis.created_at,
            updated_at=analysis.updated_at,
            hypothesis_count=hypothesis_count,
            validated_count=validated_count,
            rejected_count=rejected_count
        )


class HypothesisSummary(BaseSummary):
    """Lightweight summary of a Hypothesis for list operations."""

    hypothesis_id: str = Field(description="Unique hypothesis ID")
    statement: str = Field(description="The hypothesis statement")
    status: HypothesisStatus = Field(description="Current status")
    trace_count: int = Field(description="Number of unique traces")
    evidence_count: int = Field(description="Total evidence entries")
    supports_count: int = Field(description="Supporting evidence count")
    refutes_count: int = Field(description="Refuting evidence count")

    def get_id(self) -> str:
        """Get the unique identifier for this hypothesis."""
        return self.hypothesis_id

    @classmethod
    def from_hypothesis(cls, hypothesis: Hypothesis) -> HypothesisSummary:
        """Create a summary from a Hypothesis instance."""
        return cls(
            hypothesis_id=hypothesis.hypothesis_id,
            statement=hypothesis.statement,
            status=hypothesis.status,
            trace_count=hypothesis.trace_count,
            evidence_count=hypothesis.evidence_count,
            supports_count=hypothesis.supports_count,
            refutes_count=hypothesis.refutes_count,
            created_at=hypothesis.created_at,
            updated_at=hypothesis.updated_at
        )


class IssueSummary(BaseSummary):
    """Lightweight summary of an Issue for list operations."""

    issue_id: str = Field(description="Unique issue ID")
    title: str = Field(description="Issue title")
    severity: IssueSeverity = Field(description="Issue severity")
    status: IssueStatus = Field(description="Current status")
    trace_count: int = Field(description="Number of unique traces")
    source_run_id: str = Field(description="Run where issue was discovered")

    def get_id(self) -> str:
        """Get the unique identifier for this issue."""
        return self.issue_id

    @classmethod
    def from_issue(cls, issue: Issue) -> IssueSummary:
        """Create a summary from an Issue instance."""
        return cls(
            issue_id=issue.issue_id,
            title=issue.title,
            severity=issue.severity,
            status=issue.status,
            trace_count=issue.trace_count,
            source_run_id=issue.source_run_id,
            created_at=issue.created_at,
            updated_at=issue.updated_at
        )