from __future__ import annotations

from dataclasses import dataclass

from opentelemetry import trace as trace_api


@dataclass
class SpanStatus:
    """
    Status of the span or the trace.

    Args:
        status_code: The status code of the span or the trace.
        description: Description of the status. Optional.
    """

    status_code: StatusCode
    description: str = ""

    class StatusCode:
        # Importing here to avoid circular imports
        from mlflow.entities.trace_status import TraceStatus

        UNSPECIFIED = TraceStatus.to_string(TraceStatus.UNSPECIFIED)
        OK = TraceStatus.to_string(TraceStatus.OK)
        ERROR = TraceStatus.to_string(TraceStatus.ERROR)

    @staticmethod
    def to_otel_status(status: SpanStatus) -> trace_api.Status:
        """Convert our status object to OpenTelemetry status object."""
        if status.status_code == SpanStatus.StatusCode.OK:
            status_code = trace_api.StatusCode.OK
        elif status_code == SpanStatus.StatusCode.ERROR:
            status_code = trace_api.StatusCode.ERROR
        else:
            status_code = trace_api.StatusCode.UNSET
        return trace_api.Status(status_code, status.description)

    @staticmethod
    def from_otel_status(otel_status: trace_api.Status) -> SpanStatus:
        """Convert OpenTelemetry status object to our status object."""
        if otel_status.status_code == trace_api.StatusCode.OK:
            return SpanStatus(SpanStatus.StatusCode.OK, otel_status.description)
        elif otel_status.status_code == trace_api.StatusCode.ERROR:
            return SpanStatus(SpanStatus.StatusCode.ERROR, otel_status.description)
        else:
            return SpanStatus(SpanStatus.StatusCode.UNSPECIFIED, otel_status.description)
