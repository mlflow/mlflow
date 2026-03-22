"""
OTLP trace archival helpers for serializing/deserializing span data as TracesData protobuf.

Used by ArtifactRepository.upload_trace_data / download_trace_data (with
spans_location=ARCHIVE_REPO) when archiving or loading trace span content
(traces.pb). Keeps OTel-specific logic separate from the artifact layer.
"""

from __future__ import annotations

from opentelemetry.proto.trace.v1.trace_pb2 import TracesData

from mlflow.entities.span import Span
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracing.utils.otlp import resource_to_otel_proto

# Artifact path and filename for archived trace span data (OTLP protobuf binary).
TRACE_ARCHIVAL_ARTIFACT_PATH = "artifacts"
TRACE_ARCHIVAL_FILENAME = "traces.pb"


def spans_to_traces_data_pb(spans: list[Span]) -> bytes:
    """
    Serialize a list of spans to OTLP TracesData protobuf binary.

    Uses a single ResourceSpans with one ScopeSpans, matching the structure used
    for OTLP export. All spans must belong to the same trace.

    Args:
        spans: List of MLflow Span entities (same trace).

    Returns:
        Serialized TracesData message as bytes.

    Raises:
        MlflowException: If spans belong to more than one trace (multiple request IDs).
    """
    if not spans:
        return TracesData().SerializeToString()

    trace_ids = {span.request_id for span in spans}
    if len(trace_ids) > 1:
        raise MlflowException(
            f"All spans must belong to the same trace, but found {len(trace_ids)} "
            f"distinct request IDs: {sorted(trace_ids)}",
            error_code=INVALID_PARAMETER_VALUE,
        )

    traces_data = TracesData()
    resource_spans = traces_data.resource_spans.add()
    resource = getattr(spans[0]._span, "resource", None)
    resource_spans.resource.CopyFrom(resource_to_otel_proto(resource))
    scope_spans = resource_spans.scope_spans.add()
    scope_spans.spans.extend(span.to_otel_proto() for span in spans)
    return traces_data.SerializeToString()


def traces_data_pb_to_spans(data: bytes) -> list[Span]:
    """
    Deserialize OTLP TracesData protobuf binary to a list of MLflow Spans.

    Args:
        data: Serialized TracesData message (e.g. from traces.pb).

    Returns:
        List of MLflow Span entities.
    """
    traces_data = TracesData()
    traces_data.ParseFromString(data)
    return [
        Span.from_otel_proto(otel_span)
        for resource_spans in traces_data.resource_spans
        for scope_spans in resource_spans.scope_spans
        for otel_span in scope_spans.spans
    ]
