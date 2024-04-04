from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from opentelemetry import trace as trace_api

from mlflow.entities.trace_status import TraceStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


@dataclass
class SpanStatus:
    """
    Status of the span or the trace.

    Args:
        status_code: The status code of the span or the trace. This must be one of the
            values of the :py:class:`mlflow.entities.TraceStatus` enum or a string
            representation of it like "OK", "ERROR".
        description: Description of the status. This should be only set when the status
            is ERROR, otherwise it will be ignored.
    """

    status_code: TraceStatus
    description: str = ""

    # These class variables will not be serialized.
    _otel_status_code_to_mlflow: ClassVar = {
        trace_api.StatusCode.OK: TraceStatus.OK,
        trace_api.StatusCode.ERROR: TraceStatus.ERROR,
        trace_api.StatusCode.UNSET: TraceStatus.UNSPECIFIED,
    }
    _mlflow_status_code_to_otel: ClassVar = {
        value: key for key, value in _otel_status_code_to_mlflow.items()
    }

    def __post_init__(self):
        """
        If user provides a string status code, validate it and convert to
        the corresponding enum value.
        """
        if isinstance(self.status_code, str):
            try:
                self.status_code = TraceStatus(self.status_code)
            except ValueError:
                raise MlflowException(
                    f"{self.status_code} is not a valid TraceStatus value. "
                    f"Please use one of {[status.value for status in TraceStatus]}",
                    error_code=INVALID_PARAMETER_VALUE,
                )

    def to_otel_status(self) -> trace_api.Status:
        """
        Convert :py:class:`mlflow.entities.SpanStatus` object to OpenTelemetry status object.

        :meta private:
        """
        if self.status_code not in SpanStatus._mlflow_status_code_to_otel:
            raise MlflowException(
                f"Invalid status code: {self.status_code}", error_code=INVALID_PARAMETER_VALUE
            )

        status_code = SpanStatus._mlflow_status_code_to_otel[self.status_code]
        return trace_api.Status(status_code, self.description)

    @classmethod
    def from_otel_status(cls, otel_status: trace_api.Status) -> SpanStatus:
        """
        Convert OpenTelemetry status object to our status object.

        :meta private:
        """
        if otel_status.status_code not in cls._otel_status_code_to_mlflow:
            raise MlflowException(
                f"Got invalid status code from OpenTelemetry: {otel_status.status_code}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        mlflow_status_code = cls._otel_status_code_to_mlflow[otel_status.status_code]
        return cls(mlflow_status_code, otel_status.description)
