"""
MLflow Insights entities.

Main entity models for the MLflow Insights agent that analyzes traces to discover
and document issues with AI agents. This file contains Census-related entities.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from mlflow.insights.models.base import DatetimeFieldsMixin, SerializableModel


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
            ...     quality_metrics=quality_metrics,
            ...     additional_metadata={"experiment_ids": [1, 2, 3]},
            ... )
        """
        metadata = CensusMetadata(
            created_at=datetime.now(timezone.utc),
            table_name=table_name,
            additional_metadata=additional_metadata or {},
        )
        return cls(
            metadata=metadata,
            operational_metrics=operational_metrics,
            quality_metrics=quality_metrics,
        )
