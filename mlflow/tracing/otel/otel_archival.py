"""
Helpers for serializing archived trace spans as OTLP ``TracesData`` protobuf.
"""

from __future__ import annotations

from opentelemetry.proto.trace.v1.trace_pb2 import TracesData

from mlflow.entities.span import Span
from mlflow.exceptions import MlflowException
from mlflow.tracing.utils.otlp import resource_to_otel_proto

TRACE_ARCHIVAL_ARTIFACT_PATH = "artifacts"
TRACE_ARCHIVAL_FILENAME = "traces.pb"


def spans_to_traces_data_pb(spans: list[Span]) -> bytes:
    """
    Serialize MLflow spans to OTLP ``TracesData`` protobuf bytes.

    Archived trace payloads must be non-empty and all spans must belong to the
    same underlying OTLP trace.
    """
    if not spans:
        raise MlflowException.invalid_parameter_value(
            "Archived trace payload must include at least one span."
        )

    otlp_trace_ids = {span._trace_id for span in spans}
    if len(otlp_trace_ids) > 1:
        raise MlflowException.invalid_parameter_value(
            "Archived spans must belong to a single OTLP trace, but got distinct trace IDs: "
            f"{sorted(otlp_trace_ids)}"
        )

    traces_data = TracesData()
    resource_spans = traces_data.resource_spans.add()
    resource = getattr(spans[0]._span, "resource", None)
    resource_spans.resource.CopyFrom(resource_to_otel_proto(resource))
    scope_spans = resource_spans.scope_spans.add()
    scope_spans.spans.extend(span.to_otel_proto() for span in spans)
    return traces_data.SerializeToString()


def traces_data_pb_to_spans(data: bytes) -> list[Span]:
    """Deserialize OTLP ``TracesData`` protobuf bytes into MLflow spans."""
    if not data:
        raise MlflowException.invalid_parameter_value(
            "Archived trace payload must be a non-empty OTLP TracesData protobuf."
        )

    traces_data = TracesData()
    traces_data.ParseFromString(data)
    return [
        Span.from_otel_proto(otel_span)
        for resource_spans in traces_data.resource_spans
        for scope_spans in resource_spans.scope_spans
        for otel_span in scope_spans.spans
    ]
