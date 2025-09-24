"""
MLflow Insights entities.

Main entity models for the MLflow Insights agent that analyzes traces to discover
and document issues with AI agents. Supports AI-guided investigation workflows
with hypotheses, evidence collection, and issue tracking.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from mlflow.exceptions import MlflowException
from mlflow.insights.constants import AnalysisStatus, HypothesisStatus, IssueSeverity, IssueStatus
from mlflow.insights.models.base import (
    DatetimeFieldsMixin,
    EvidencedModel,
    EvidenceEntry,
    ExtensibleModel,
    SerializableModel,
    TimestampedModel,
)
from mlflow.insights.utils import normalize_evidence


class Analysis(SerializableModel, TimestampedModel, ExtensibleModel):
    """
    Investigation run for analyzing MLflow traces for issues.

    An Analysis represents an AI-guided investigation run that examines
    existing traces for operational issues (latency, errors, authentication)
    and quality issues (formatting, response quality). Users can provide
    focus areas and guidance to steer the investigation.
    Stored as 'analysis.yaml' in the MLflow run artifacts.
    """

    name: str = Field(description="Human-readable name for the investigation")
    description: str = Field(
        description="Description of investigation focus areas and guidance provided by the user"
    )
    status: AnalysisStatus = Field(
        default=AnalysisStatus.ACTIVE, description="Current status of the analysis"
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
    statement: str = Field(
        description="The hypothesis being tested "
        "(e.g., 'The agent exhibits high latency when the query_sql tool is used')"
    )
    testing_plan: str = Field(
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
    source_run_id: str = Field(description="MLflow analysis run ID where this issue was discovered")
    hypothesis_id: str | None = Field(
        default=None,
        description="Source hypothesis ID if this issue was derived from a validated hypothesis",
    )
    title: str = Field(description="Brief, descriptive title for the issue")
    description: str = Field(description="Detailed description of the problem and its impact")
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


class ErrorSpanSummary(BaseModel, DatetimeFieldsMixin):
    """Summary of an error span for reporting."""

    span: str = Field(description="Name of the span with errors")
    count: int = Field(description="Number of errors for this span")


class SlowToolSummary(BaseModel, DatetimeFieldsMixin):
    """Summary of a slow tool for reporting."""

    tool: str = Field(description="Name of the tool span")
    p95_ms: float = Field(description="95th percentile latency in milliseconds")


class ErrorSummary(BaseModel, DatetimeFieldsMixin):
    """Summary of error metrics."""

    total_errors: int = Field(description="Total number of errors")
    error_rate: float = Field(description="Overall error rate percentage")
    top_error_spans: list[ErrorSpanSummary] = Field(
        default_factory=list, description="Top error spans"
    )


class PerformanceSummary(BaseModel, DatetimeFieldsMixin):
    """Summary of performance metrics."""

    total_traces: int = Field(description="Total number of traces")
    p50_latency_ms: float = Field(description="50th percentile latency")
    p95_latency_ms: float = Field(description="95th percentile latency")
    p99_latency_ms: float = Field(description="99th percentile latency")
    slowest_tools: list[SlowToolSummary] = Field(
        default_factory=list, description="Slowest performing tools"
    )


class QualitySummary(BaseModel, DatetimeFieldsMixin):
    """Summary of quality metrics."""

    minimal_responses: float = Field(description="Percentage of minimal responses")
    response_quality_issues: float = Field(description="Percentage of quality issues")
    rushed_processing: float = Field(description="Percentage of rushed processing")
    verbosity: float = Field(description="Percentage of verbose responses")


class TimeRange(BaseModel, DatetimeFieldsMixin):
    """Time range for census data."""

    start: datetime = Field(description="Start timestamp")
    end: datetime = Field(description="End timestamp")

    @field_validator("start", "end", mode="before")
    @classmethod
    def validate_datetime(cls, v):
        return cls.parse_datetime(v)


class TimeBucket(BaseModel, DatetimeFieldsMixin):
    """Time-bucketed metrics for operational analysis."""

    time_bucket: datetime = Field(description="Time bucket timestamp")
    total_traces: int = Field(description="Total trace count in bucket")
    ok_count: int = Field(description="Successful trace count")
    error_count: int = Field(description="Error trace count")
    error_rate: float = Field(description="Error rate percentage")
    p95_latency_ms: float = Field(description="95th percentile latency in milliseconds")

    @field_validator("time_bucket", mode="before")
    @classmethod
    def validate_time_bucket(cls, v):
        return cls.parse_datetime(v)


class ErrorSpan(BaseModel, DatetimeFieldsMixin):
    """Error span statistics with sample traces."""

    error_span_name: str = Field(description="Name of the span with errors")
    count: int = Field(description="Number of errors for this span")
    pct_of_errors: float = Field(description="Percentage of total errors")
    sample_trace_ids: list[str] = Field(
        default_factory=list, description="Sample trace IDs showing this error"
    )


class SlowTool(BaseModel, DatetimeFieldsMixin):
    """Tool performance statistics."""

    tool_span_name: str = Field(description="Name of the tool span")
    count: int = Field(description="Number of invocations")
    median_latency_ms: float = Field(description="Median latency in milliseconds")
    p95_latency_ms: float = Field(description="95th percentile latency in milliseconds")
    sample_trace_ids: list[str] = Field(
        default_factory=list, description="Sample trace IDs for this tool"
    )


class OperationalMetrics(BaseModel, DatetimeFieldsMixin):
    """System performance, errors, and latency metrics."""

    total_traces: int = Field(description="Total number of traces analyzed")
    ok_count: int = Field(description="Count of successful traces")
    error_count: int = Field(description="Count of error traces")
    error_rate: float = Field(description="Overall error rate percentage")
    first_trace_timestamp: datetime = Field(description="Timestamp of earliest trace")
    last_trace_timestamp: datetime = Field(description="Timestamp of latest trace")
    max_latency_ms: float = Field(description="Maximum latency in milliseconds")

    @field_validator("first_trace_timestamp", "last_trace_timestamp", mode="before")
    @classmethod
    def validate_timestamps(cls, v):
        return cls.parse_datetime(v)

    p50_latency_ms: float = Field(description="50th percentile latency")
    p90_latency_ms: float = Field(description="90th percentile latency")
    p95_latency_ms: float = Field(description="95th percentile latency")
    p99_latency_ms: float = Field(description="99th percentile latency")
    time_buckets: list[TimeBucket] = Field(
        default_factory=list, description="Time-bucketed performance metrics"
    )
    top_error_spans: list[ErrorSpan] = Field(
        default_factory=list, description="Spans with most errors"
    )
    top_slow_tools: list[SlowTool] = Field(
        default_factory=list, description="Slowest performing tools"
    )
    error_sample_trace_ids: list[str] = Field(
        default_factory=list, description="Sample trace IDs with errors"
    )


class QualityMetric(BaseModel, DatetimeFieldsMixin):
    """Individual quality metric with samples."""

    value: float = Field(description="Metric value as percentage")
    description: str = Field(description="Description of what this metric measures")
    sample_trace_ids: list[str] = Field(
        default_factory=list, description="Sample trace IDs exhibiting this quality issue"
    )


class QualityMetrics(BaseModel, DatetimeFieldsMixin):
    """Agent response quality analysis metrics."""

    minimal_responses: QualityMetric = Field(description="Analysis of minimal/incomplete responses")
    response_quality_issues: QualityMetric = Field(
        description="Analysis of responses with quality problems"
    )
    rushed_processing: QualityMetric = Field(
        description="Analysis of complex requests processed too quickly"
    )
    verbosity: QualityMetric = Field(description="Analysis of overly verbose responses")


class CensusMetadata(BaseModel, DatetimeFieldsMixin):
    """Metadata about the census analysis."""

    created_at: datetime = Field(description="Timestamp when census was created")
    table_name: str = Field(description="Source table name")
    additional_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional custom metadata"
    )

    @field_validator("created_at", mode="before")
    @classmethod
    def validate_created_at(cls, v):
        """Convert string to datetime if needed."""
        return cls.parse_datetime(v)


class Census(SerializableModel, DatetimeFieldsMixin):
    """
    Comprehensive baseline census of trace data.

    Provides operational and quality metrics for a collection of traces,
    useful for establishing baselines and detecting anomalies.
    """

    metadata: CensusMetadata = Field(description="Census metadata and source information")
    operational_metrics: OperationalMetrics = Field(
        description="System performance and error metrics"
    )
    quality_metrics: QualityMetrics = Field(description="Response quality analysis")

    @classmethod
    def create_with_timestamp(
        cls,
        table_name: str,
        operational_metrics: OperationalMetrics,
        quality_metrics: QualityMetrics,
        additional_metadata: dict[str, Any] | None = None,
    ) -> "Census":
        """
        Create a new census with automatically set current timestamp.

        This is a convenience factory method that sets the creation timestamp
        to the current time and properly structures the metadata. Use the standard
        Pydantic constructor if you need to specify a different timestamp.

        Args:
            table_name: Source table name
            operational_metrics: Operational performance metrics
            quality_metrics: Response quality metrics
            additional_metadata: Optional additional metadata

        Returns:
            New Census instance with current timestamp

        Example:
            >>> census = Census.create_with_timestamp(
            ...     table_name="traces_table",
            ...     operational_metrics=op_metrics,
            ...     quality_metrics=qual_metrics,
            ... )
        """
        return cls(
            metadata=CensusMetadata(
                created_at=datetime.now(),
                table_name=table_name,
                additional_metadata=additional_metadata or {},
            ),
            operational_metrics=operational_metrics,
            quality_metrics=quality_metrics,
        )

    def get_error_summary(self) -> ErrorSummary:
        """Get a summary of error metrics."""
        return ErrorSummary(
            total_errors=self.operational_metrics.error_count,
            error_rate=self.operational_metrics.error_rate,
            top_error_spans=[
                ErrorSpanSummary(span=span.error_span_name, count=span.count)
                for span in self.operational_metrics.top_error_spans[:5]
            ],
        )

    def get_performance_summary(self) -> PerformanceSummary:
        """Get a summary of performance metrics."""
        return PerformanceSummary(
            total_traces=self.operational_metrics.total_traces,
            p50_latency_ms=self.operational_metrics.p50_latency_ms,
            p95_latency_ms=self.operational_metrics.p95_latency_ms,
            p99_latency_ms=self.operational_metrics.p99_latency_ms,
            slowest_tools=[
                SlowToolSummary(tool=tool.tool_span_name, p95_ms=tool.p95_latency_ms)
                for tool in self.operational_metrics.top_slow_tools[:5]
            ],
        )

    def get_quality_summary(self) -> QualitySummary:
        """Get a summary of quality metrics."""
        return QualitySummary(
            minimal_responses=self.quality_metrics.minimal_responses.value,
            response_quality_issues=self.quality_metrics.response_quality_issues.value,
            rushed_processing=self.quality_metrics.rushed_processing.value,
            verbosity=self.quality_metrics.verbosity.value,
        )

    def has_quality_issues(self, threshold: float = 10.0) -> bool:
        """
        Check if any quality metric exceeds the threshold.

        Args:
            threshold: Percentage threshold for quality issues

        Returns:
            True if any quality metric exceeds threshold
        """
        return any(
            metric.value > threshold
            for metric in [
                self.quality_metrics.minimal_responses,
                self.quality_metrics.response_quality_issues,
                self.quality_metrics.rushed_processing,
                self.quality_metrics.verbosity,
            ]
        )

    def get_time_range(self) -> TimeRange:
        """Get the time range covered by this census."""
        return TimeRange(
            start=self.operational_metrics.first_trace_timestamp,
            end=self.operational_metrics.last_trace_timestamp,
        )
