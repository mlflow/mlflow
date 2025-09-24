"""Entity models for traffic and tool metrics."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject


@dataclass
class TrafficSummary(_MlflowObject):
    """
    Summary statistics for traffic metrics.

    Attributes:
        request_count: Total number of requests
        request_rate_p50_qpm: 50th percentile request rate (queries per minute)
        request_rate_p90_qpm: 90th percentile request rate (queries per minute)
        request_rate_p99_qpm: 99th percentile request rate (queries per minute)
    """

    request_count: int
    request_rate_p50_qpm: float | None
    request_rate_p90_qpm: float | None
    request_rate_p99_qpm: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "request_rate_p50_qpm": self.request_rate_p50_qpm,
            "request_rate_p90_qpm": self.request_rate_p90_qpm,
            "request_rate_p99_qpm": self.request_rate_p99_qpm,
        }


@dataclass
class TrafficTimePoint(_MlflowObject):
    """
    A single point in time for traffic metrics.

    Attributes:
        timestamp_millis: Timestamp in milliseconds since epoch
        request_count: Number of requests at this time point
        request_rate_qpm: Request rate at this time point (queries per minute)
    """

    timestamp_millis: int
    request_count: int
    request_rate_qpm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_millis": self.timestamp_millis,
            "request_count": self.request_count,
            "request_rate_qpm": self.request_rate_qpm,
        }


@dataclass
class TrafficVolume(_MlflowObject):
    """
    Traffic volume metrics with summary and time series.

    Attributes:
        summary: Summary statistics for traffic
        time_series: List of time points for traffic metrics
    """

    summary: TrafficSummary
    time_series: list[TrafficTimePoint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "time_series": [tp.to_dict() for tp in self.time_series],
        }


@dataclass
class LatencySummary(_MlflowObject):
    """
    Summary statistics for latency metrics.

    Attributes:
        p50_latency_millis: 50th percentile latency in milliseconds
        p90_latency_millis: 90th percentile latency in milliseconds
        p99_latency_millis: 99th percentile latency in milliseconds
        mean_latency_millis: Mean latency in milliseconds
    """

    p50_latency_millis: float | None
    p90_latency_millis: float | None
    p99_latency_millis: float | None
    mean_latency_millis: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "p50_latency_millis": self.p50_latency_millis,
            "p90_latency_millis": self.p90_latency_millis,
            "p99_latency_millis": self.p99_latency_millis,
            "mean_latency_millis": self.mean_latency_millis,
        }


@dataclass
class LatencyTimePoint(_MlflowObject):
    """
    A single point in time for latency metrics.

    Attributes:
        timestamp_millis: Timestamp in milliseconds since epoch
        p50_latency_millis: 50th percentile latency at this time point
        p90_latency_millis: 90th percentile latency at this time point
        p99_latency_millis: 99th percentile latency at this time point
        mean_latency_millis: Mean latency at this time point
    """

    timestamp_millis: int
    p50_latency_millis: float | None
    p90_latency_millis: float | None
    p99_latency_millis: float | None
    mean_latency_millis: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_millis": self.timestamp_millis,
            "p50_latency_millis": self.p50_latency_millis,
            "p90_latency_millis": self.p90_latency_millis,
            "p99_latency_millis": self.p99_latency_millis,
            "mean_latency_millis": self.mean_latency_millis,
        }


@dataclass
class TrafficLatency(_MlflowObject):
    """
    Traffic latency metrics with summary and time series.

    Attributes:
        summary: Summary statistics for latency
        time_series: List of time points for latency metrics
    """

    summary: LatencySummary
    time_series: list[LatencyTimePoint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "time_series": [tp.to_dict() for tp in self.time_series],
        }


@dataclass
class ToolMetric(_MlflowObject):
    """
    Metrics for a single tool.

    Attributes:
        tool_name: Name of the tool
        usage_count: Number of times the tool was used
        p50_latency_millis: 50th percentile latency
        p90_latency_millis: 90th percentile latency
        p99_latency_millis: 99th percentile latency
        mean_latency_millis: Mean latency
        error_rate: Error rate (0.0 to 1.0)
    """

    tool_name: str
    usage_count: int
    p50_latency_millis: float | None
    p90_latency_millis: float | None
    p99_latency_millis: float | None
    mean_latency_millis: float | None
    error_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "usage_count": self.usage_count,
            "p50_latency_millis": self.p50_latency_millis,
            "p90_latency_millis": self.p90_latency_millis,
            "p99_latency_millis": self.p99_latency_millis,
            "mean_latency_millis": self.mean_latency_millis,
            "error_rate": self.error_rate,
        }


@dataclass
class ToolMetricTimePoint(_MlflowObject):
    """
    A single point in time for tool metrics.

    Attributes:
        timestamp_millis: Timestamp in milliseconds since epoch
        metrics: List of tool metrics at this time point
    """

    timestamp_millis: int
    metrics: list[ToolMetric]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_millis": self.timestamp_millis,
            "metrics": [m.to_dict() for m in self.metrics],
        }


@dataclass
class ToolMetrics(_MlflowObject):
    """
    Tool usage metrics with summary and time series.

    Attributes:
        summary: List of tool metrics summarizing the entire period
        time_series: List of time points for tool metrics
    """

    summary: list[ToolMetric]
    time_series: list[ToolMetricTimePoint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": [m.to_dict() for m in self.summary],
            "time_series": [tp.to_dict() for tp in self.time_series],
        }