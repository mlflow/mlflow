from __future__ import annotations

from dataclasses import dataclass

from opentelemetry import trace as trace_api

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracing.types.constant import TraceStatusCode


@dataclass
class SpanStatus:
    """
    Status of the span or the trace.

    Args:
        status_code: The status code of the span or the trace.
        description: Description of the status. Optional.
    """

    status_code: TraceStatusCode
    description: str = ""

    def to_otel_status(self) -> trace_api.Status:
        """Convert our status object to OpenTelemetry status object."""
        if self.status_code == TraceStatusCode.OK:
            status_code = trace_api.StatusCode.OK
        elif self.status_code == TraceStatusCode.ERROR:
            status_code = trace_api.StatusCode.ERROR
        elif self.status_code == TraceStatusCode.UNSPECIFIED:
            status_code = trace_api.StatusCode.UNSET
        else:
            raise MlflowException(
                f"Invalid status code: {self.status_code}", error_code=INVALID_PARAMETER_VALUE
            )
        return trace_api.Status(status_code, self.description)

    @classmethod
    def from_otel_status(cls, otel_status: trace_api.Status) -> SpanStatus:
        """Convert OpenTelemetry status object to our status object."""
        if otel_status.status_code == trace_api.StatusCode.OK:
            return cls(TraceStatusCode.OK, otel_status.description)
        elif otel_status.status_code == trace_api.StatusCode.ERROR:
            return cls(TraceStatusCode.ERROR, otel_status.description)
        elif otel_status.status_code == trace_api.StatusCode.UNSET:
            return cls(TraceStatusCode.UNSPECIFIED, otel_status.description)
        else:
            raise MlflowException(
                f"Got invalid status code from OpenTelemetry: {otel_status.status_code}",
                error_code=INVALID_PARAMETER_VALUE,
            )
