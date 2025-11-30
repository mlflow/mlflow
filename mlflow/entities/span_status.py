from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from opentelemetry import trace as trace_api
from opentelemetry.proto.trace.v1.trace_pb2 import Status as OtelStatus

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class SpanStatusCode(str, Enum):
    """Enum for status code of a span"""

    # Uses the same set of status codes as OpenTelemetry
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"

    def to_otel_proto_status_code_name(self) -> str:
        """
        Convert the SpanStatusCode to the corresponding OpenTelemetry protobuf enum name.
        """
        proto_code = OtelStatus.StatusCode
        mapping = {
            SpanStatusCode.UNSET: proto_code.Name(proto_code.STATUS_CODE_UNSET),
            SpanStatusCode.OK: proto_code.Name(proto_code.STATUS_CODE_OK),
            SpanStatusCode.ERROR: proto_code.Name(proto_code.STATUS_CODE_ERROR),
        }
        return mapping[self]

    @staticmethod
    def from_otel_proto_status_code_name(status_code_name: str) -> SpanStatusCode:
        """
        Convert an OpenTelemetry protobuf enum name to the corresponding SpanStatusCode enum value.
        """
        proto_code = OtelStatus.StatusCode
        mapping = {
            proto_code.Name(proto_code.STATUS_CODE_UNSET): SpanStatusCode.UNSET,
            proto_code.Name(proto_code.STATUS_CODE_OK): SpanStatusCode.OK,
            proto_code.Name(proto_code.STATUS_CODE_ERROR): SpanStatusCode.ERROR,
        }
        try:
            return mapping[status_code_name]
        except KeyError:
            raise MlflowException(
                f"Invalid status code name: {status_code_name}. "
                f"Valid values are: {', '.join(mapping.keys())}",
                error_code=INVALID_PARAMETER_VALUE,
            )


@dataclass
class SpanStatus:
    """
    Status of the span or the trace.

    Args:
        status_code: The status code of the span or the trace. This must be one of the
            values of the :py:class:`mlflow.entities.SpanStatusCode` enum or a string
            representation of it like "OK", "ERROR".
        description: Description of the status. This should be only set when the status
            is ERROR, otherwise it will be ignored.
    """

    status_code: SpanStatusCode
    description: str = ""

    def __post_init__(self):
        """
        If user provides a string status code, validate it and convert to
        the corresponding enum value.
        """
        if isinstance(self.status_code, str):
            try:
                self.status_code = SpanStatusCode(self.status_code)
            except ValueError:
                raise MlflowException(
                    f"{self.status_code} is not a valid SpanStatusCode value. "
                    f"Please use one of {[status_code.value for status_code in SpanStatusCode]}",
                    error_code=INVALID_PARAMETER_VALUE,
                )

    def to_otel_status(self) -> trace_api.Status:
        """
        Convert :py:class:`mlflow.entities.SpanStatus` object to OpenTelemetry status object.

        :meta private:
        """
        try:
            status_code = getattr(trace_api.StatusCode, self.status_code.name)
        except AttributeError:
            raise MlflowException(
                f"Invalid status code: {self.status_code}", error_code=INVALID_PARAMETER_VALUE
            )
        return trace_api.Status(status_code, self.description)

    @classmethod
    def from_otel_status(cls, otel_status: trace_api.Status) -> SpanStatus:
        """
        Convert OpenTelemetry status object to our status object.

        :meta private:
        """
        try:
            status_code = SpanStatusCode(otel_status.status_code.name)
        except ValueError:
            raise MlflowException(
                f"Got invalid status code from OpenTelemetry: {otel_status.status_code}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return cls(status_code, otel_status.description or "")

    def to_otel_proto_status(self):
        """
        Convert to OpenTelemetry protobuf Status for OTLP export.

        :meta private:
        """
        status = OtelStatus()
        if self.status_code == SpanStatusCode.OK:
            status.code = OtelStatus.StatusCode.STATUS_CODE_OK
        elif self.status_code == SpanStatusCode.ERROR:
            status.code = OtelStatus.StatusCode.STATUS_CODE_ERROR
        else:
            status.code = OtelStatus.StatusCode.STATUS_CODE_UNSET

        if self.description:
            status.message = self.description

        return status

    @classmethod
    def from_otel_proto_status(cls, otel_proto_status) -> SpanStatus:
        """
        Create a SpanStatus from an OpenTelemetry protobuf Status.

        :meta private:
        """
        # Map protobuf status codes to SpanStatusCode
        if otel_proto_status.code == OtelStatus.STATUS_CODE_OK:
            status_code = SpanStatusCode.OK
        elif otel_proto_status.code == OtelStatus.STATUS_CODE_ERROR:
            status_code = SpanStatusCode.ERROR
        else:
            status_code = SpanStatusCode.UNSET

        return cls(status_code, otel_proto_status.message or "")
