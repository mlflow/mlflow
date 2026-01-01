"""Session-level online scoring processor for executing scorers on completed sessions."""

from dataclasses import dataclass


@dataclass
class CompletedSession:
    """
    Metadata about a session that has been determined complete and is eligible for online scoring.

    Contains only the session ID and timestamp range, not the actual trace data.
    """

    session_id: str
    first_trace_timestamp_ms: int
    last_trace_timestamp_ms: int
