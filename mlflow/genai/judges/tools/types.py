"""
Type definitions for MLflow GenAI judge tools.

This module contains dataclass definitions for results and other types
used across the judge tools system.
"""

from dataclasses import dataclass

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
