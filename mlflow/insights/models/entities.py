"""
MLflow Insights entities.

Main entity models for the MLflow Insights agent that analyzes traces to discover
and document issues with AI agents. Supports AI-guided investigation workflows
with hypotheses, evidence collection, and issue tracking.

Entity Architecture:
====================

This module defines the three core entities of the Insights system:

1. **Analysis**: The top-level investigation container
   - Represents a user-initiated investigation session
   - Tracks the overall investigation status and metadata
   - Contains focus areas and guidance from the user
   - Stored as 'analysis.yaml' in the MLflow run artifacts

2. **Hypothesis**: Testable theories about agent problems
   - Generated during analysis to explore potential issues
   - Can be validated or rejected based on evidence
   - Tracks supporting and refuting evidence from traces
   - Stored as 'hypothesis_<id>.yaml' in the MLflow run artifacts

3. **Issue**: Confirmed problems requiring attention
   - Created from validated hypotheses
   - Filed against the experiment for cross-run visibility
   - Can be consumed by external tools (CLI/MCP) for automated fixes
   - Tracks resolution status and related assessments
   - Stored as 'issue_<id>.yaml' in the experiment artifacts

Validation Strategy:
--------------------
All required string fields use Pydantic's AfterValidator with type aliases to ensure:
- Non-empty values (no whitespace-only strings)
- Consistent trimming of leading/trailing whitespace
- Clear error messages identifying the specific field

Evidence Model:
---------------
Both Hypotheses and Issues collect evidence from traces:
- Hypotheses track whether evidence supports or refutes the theory
- Issues only track demonstrative evidence (no support/refute concept)
- Multiple evidence entries can reference the same trace for different insights

Status Workflow:
----------------
- Analysis: ACTIVE -> COMPLETED/ARCHIVED/ERROR
- Hypothesis: TESTING -> VALIDATED/REJECTED/ERROR
- Issue: OPEN -> IN_PROGRESS -> RESOLVED/REJECTED/ERROR

Integration Points:
-------------------
- Storage: YAML files in MLflow artifacts for portability
- Discovery: Issues filed at experiment level for visibility
- Automation: Issues consumable by coding agents for fixes
- Extensibility: Metadata fields for future enhancements
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import uuid4

from pydantic import AfterValidator, Field, field_validator

from mlflow.exceptions import MlflowException
from mlflow.insights.constants import AnalysisStatus, HypothesisStatus, IssueSeverity, IssueStatus
from mlflow.insights.models.base import (
    EvidencedModel,
    EvidenceEntry,
    ExtensibleModel,
    SerializableModel,
    TimestampedModel,
)
from mlflow.insights.utils import normalize_evidence, validate_non_empty_string

NonEmptyAnalysisName = Annotated[
    str, AfterValidator(lambda v: validate_non_empty_string(v, "Analysis name"))
]
NonEmptyAnalysisDescription = Annotated[
    str, AfterValidator(lambda v: validate_non_empty_string(v, "Analysis description"))
]
NonEmptyHypothesisStatement = Annotated[
    str, AfterValidator(lambda v: validate_non_empty_string(v, "Hypothesis statement"))
]
NonEmptyHypothesisTestingPlan = Annotated[
    str, AfterValidator(lambda v: validate_non_empty_string(v, "Hypothesis testing plan"))
]
NonEmptyIssueSourceRunId = Annotated[
    str, AfterValidator(lambda v: validate_non_empty_string(v, "Issue source_run_id"))
]
NonEmptyIssueTitle = Annotated[
    str, AfterValidator(lambda v: validate_non_empty_string(v, "Issue title"))
]
NonEmptyIssueDescription = Annotated[
    str, AfterValidator(lambda v: validate_non_empty_string(v, "Issue description"))
]


class Analysis(SerializableModel, TimestampedModel, ExtensibleModel):
    """
    Investigation run for analyzing MLflow traces for issues.

    An Analysis represents an AI-guided investigation run that examines
    existing traces for operational issues (latency, errors, authentication)
    and quality issues (formatting, response quality). Users can provide
    focus areas and guidance to steer the investigation.
    Stored as 'analysis.yaml' in the MLflow run artifacts.
    """

    name: NonEmptyAnalysisName = Field(description="Human-readable name for the investigation")
    description: NonEmptyAnalysisDescription = Field(
        description="Description of investigation focus areas and guidance provided by the user"
    )
    status: AnalysisStatus = Field(
        default=AnalysisStatus.ACTIVE, description="Current status of the analysis"
    )

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

    def mark_error(self, error_message: str | None = None) -> None:
        """
        Mark the analysis as having an error.

        Args:
            error_message: Optional error description to store in metadata
        """
        self.status = AnalysisStatus.ERROR
        if error_message:
            self.metadata["error_message"] = error_message
        self.update_timestamp()


class Hypothesis(SerializableModel, TimestampedModel, ExtensibleModel, EvidencedModel):
    """
    Testable statement about potential issues with an Agent.

    A Hypothesis represents a testable theory about problems with the user's
    Agent as observed in traces. It includes a testing plan describing how
    to validate/refute the hypothesis and collects evidence from traces.
    Hypotheses can be validated or invalidated during the investigation.
    Stored as 'hypothesis_<id>.yaml' in the MLflow run artifacts.
    """

    hypothesis_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier (UUID) for this hypothesis",
    )
    statement: NonEmptyHypothesisStatement = Field(
        description="The hypothesis being tested "
        "(e.g., 'The agent exhibits high latency when the query_sql tool is used')"
    )
    testing_plan: NonEmptyHypothesisTestingPlan = Field(
        description="Plan for testing the hypothesis, "
        "including how to validate/refute it by analyzing traces"
    )
    status: HypothesisStatus = Field(
        default=HypothesisStatus.TESTING, description="Current testing status"
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Quantitative metrics related to hypothesis testing"
    )

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence_field(cls, v: Any) -> list[EvidenceEntry]:
        """Normalize evidence to list of EvidenceEntry objects for hypotheses."""
        return normalize_evidence(v, for_issue=False)

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

    def mark_error(self, error_message: str | None = None) -> None:
        """
        Mark the hypothesis as having an error during testing.

        Args:
            error_message: Optional error description to store in metadata
        """
        self.status = HypothesisStatus.ERROR
        if error_message:
            self.metadata["error_message"] = error_message
        self.update_timestamp()


class Issue(SerializableModel, TimestampedModel, ExtensibleModel, EvidencedModel):
    """
    Discovered problem with an Agent derived from validated hypotheses.

    An Issue represents a confirmed problem with the Agent that was discovered
    through investigation and hypothesis validation. Issues are filed against
    the MLflow experiment (not the analysis run) and include evidence from
    traces. Users can accept or reject issues. Issues can be consumed by
    coding agents via CLI/MCP to help fix the underlying problems.
    Stored as 'issue_<id>.yaml' in the experiment artifacts.
    """

    issue_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique identifier (UUID) for this issue"
    )
    source_run_id: NonEmptyIssueSourceRunId = Field(
        description="MLflow analysis run ID where this issue was discovered"
    )
    hypothesis_id: str | None = Field(
        default=None,
        description="Source hypothesis ID if this issue was derived from a validated hypothesis",
    )
    title: NonEmptyIssueTitle = Field(description="Brief, descriptive title for the issue")
    description: NonEmptyIssueDescription = Field(
        description="Detailed description of the problem and its impact"
    )
    severity: IssueSeverity = Field(description="Severity level of the issue")
    status: IssueStatus = Field(default=IssueStatus.OPEN, description="Current issue status")
    assessments: list[str] = Field(
        default_factory=list, description="List of assessment names/IDs related to this issue"
    )
    resolution: str | None = Field(
        default=None, description="Description of how the issue was resolved (when applicable)"
    )

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence_field(cls, v: Any) -> list[EvidenceEntry]:
        """Normalize evidence to list of EvidenceEntry objects for issues."""
        return normalize_evidence(v, for_issue=True)

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
            if stripped := item.strip():
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
        if not (cleaned := assessment_name.strip() if assessment_name else ""):
            raise MlflowException.invalid_parameter_value("Assessment name cannot be empty")

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
        if not (stripped_resolution := resolution.strip() if resolution else ""):
            raise MlflowException.invalid_parameter_value("Resolution description cannot be empty")

        self.status = IssueStatus.RESOLVED
        self.resolution = stripped_resolution
        self.update_timestamp()

    def reject(self, reason: str | None = None) -> None:
        """
        Reject the issue as invalid or not actionable.

        Args:
            reason: Optional explanation for rejection
        """
        self.status = IssueStatus.REJECTED
        if reason and (stripped_reason := reason.strip()):
            self.resolution = f"Rejected: {stripped_reason}"
        self.update_timestamp()

    def reopen(self) -> None:
        """Reopen a resolved or rejected issue."""
        self.status = IssueStatus.OPEN
        self.resolution = None
        self.update_timestamp()

    def mark_error(self, error_message: str | None = None) -> None:
        """
        Mark the issue as having an error.

        Args:
            error_message: Optional error description to store in metadata
        """
        self.status = IssueStatus.ERROR
        if error_message:
            self.metadata["error_message"] = error_message
        self.update_timestamp()
