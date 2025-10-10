"""
Shared types for MLflow GenAI judge tools.

This module provides common data structures and types that can be reused
across multiple judge tools for consistent data representation.
"""

from dataclasses import dataclass
from typing import Any

from mlflow.entities.assessment import FeedbackValueType
from mlflow.entities.span_status import SpanStatus
from mlflow.entities.trace_state import TraceState
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


@experimental(version="3.5.0")
@dataclass
class Expectation:
    """Expectation for a trace."""

    name: str
    source: str
    rationale: str | None
    span_id: str | None
    assessment_id: str | None
    value: Any


@experimental(version="3.5.0")
@dataclass
class Feedback:
    """Feedback for a trace."""

    name: str
    source: str
    rationale: str | None
    span_id: str | None
    assessment_id: str | None
    value: FeedbackValueType | None
    error_code: str | None
    error_message: str | None
    stack_trace: str | None


@experimental(version="3.5.0")
@dataclass
class TraceInfo:
    """Information about a single trace."""

    trace_id: str
    request_time: int
    state: TraceState
    request: str | None
    response: str | None
    execution_duration: int | None
    assessments: list[Expectation | Feedback]


@experimental(version="3.4.0")
@dataclass
class HistoricalTrace:
    """Historical trace from the same session."""

    trace_info: Any
    request: str | None
    response: str | None
