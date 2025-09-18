"""
Summary models for MLflow Insights.

Lightweight representations of entities for list operations and UI display.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from mlflow.insights.constants import AnalysisStatus, HypothesisStatus, IssueSeverity, IssueStatus

if TYPE_CHECKING:
    from mlflow.insights.models.entities import Analysis, Hypothesis, Issue


class BaseSummary(BaseModel, ABC):
    """Base class for summary models."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    created_at: datetime = Field(description="Creation timestamp in UTC")
    updated_at: datetime = Field(description="Last update timestamp in UTC")

    @abstractmethod
    def get_id(self) -> str:
        """Get the unique identifier for this summary."""


class AnalysisSummary(BaseSummary):
    """Lightweight summary of an investigation run for list operations."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

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
        cls, run_id: str, analysis: Analysis, hypotheses: list[Hypothesis] | None = None
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
            rejected_count=rejected_count,
        )


class HypothesisSummary(BaseSummary):
    """Lightweight summary of a hypothesis for list operations."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

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
            updated_at=hypothesis.updated_at,
        )


class IssueSummary(BaseSummary):
    """Lightweight summary of a discovered issue for list operations."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

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
            updated_at=issue.updated_at,
        )
