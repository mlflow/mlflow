"""
Shared types for MLflow GenAI judge tools.

This module provides common data structures and types that can be reused
across multiple judge tools for consistent data representation.
"""

from dataclasses import dataclass

from mlflow.entities.span_status import SpanStatus
from mlflow.entities.trace_info import TraceInfo
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
@dataclass
class SpanResult:
    """Result from getting span content."""

    span_id: str | None
    content: str | None
    content_size_bytes: int
    page_token: str | None = None
    error: str | None = None


@experimental(version="3.4.0")
@dataclass
class SpanInfo:
    """Information about a single span."""

    span_id: str
    name: str
    span_type: str
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    parent_id: str | None
    status: SpanStatus
    is_root: bool
    attribute_names: list[str]


@experimental(version="3.4.0")
@dataclass
class HistoricalTrace:
    """
    Historical trace containing trace metadata and request/response data.

    Used for multi-turn evaluation to provide judges with conversation history
    from previous interactions within the same session.
    """

    trace_info: TraceInfo
    request: str | None
    response: str | None
