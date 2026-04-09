"""
Helpers for serializing archived trace spans as OTLP ``TracesData`` protobuf.
"""

from __future__ import annotations

from typing import Any

from google.protobuf.message import DecodeError
from opentelemetry.proto.trace.v1.trace_pb2 import TracesData

from mlflow.entities.span import Span
from mlflow.exceptions import MlflowException
from mlflow.tracing.utils.otlp import resource_to_otel_proto

TRACE_ARCHIVAL_FILENAME = "traces.pb"


def _sort_spans_for_trace_output(spans: list[Span]) -> list[Span]:
    """Normalize archived spans to MLflow's canonical root-first display order."""
    return sorted(
        spans,
        key=lambda span: (
            0 if span.parent_id is None else 1,
            span.start_time_ns,
            span.span_id,
        ),
    )


def normalize_otel_resource_attributes(resource) -> tuple[tuple[str, Any], ...]:
    """Convert resource attributes into an order-insensitive comparable representation."""
    if resource is None:
        return ()

    return tuple(
        (str(key), _normalize_otel_resource_attribute_value(value))
        for key, value in sorted(resource.attributes.items(), key=lambda item: str(item[0]))
    )


def _normalize_otel_resource_attribute_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (str(key), _normalize_otel_resource_attribute_value(nested_value))
            for key, nested_value in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_otel_resource_attribute_value(item) for item in value)
    return value


def spans_to_traces_data_pb(spans: list[Span]) -> bytes:
    """
    Serialize MLflow spans to OTLP ``TracesData`` protobuf bytes.

    Archived trace payloads must be non-empty and all spans must belong to the
    same underlying OTLP trace and resource.
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

    resource = getattr(spans[0]._span, "resource", None)
    normalized_resource = normalize_otel_resource_attributes(resource)
    resource_proto = resource_to_otel_proto(resource)
    if any(
        normalize_otel_resource_attributes(getattr(span._span, "resource", None))
        != normalized_resource
        for span in spans[1:]
    ):
        raise MlflowException.invalid_parameter_value(
            "Archived spans must share the same OTLP resource."
        )

    traces_data = TracesData()
    resource_spans = traces_data.resource_spans.add()
    resource_spans.resource.CopyFrom(resource_proto)
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
    try:
        traces_data.ParseFromString(data)
    except DecodeError as e:
        raise MlflowException.invalid_parameter_value(
            "Archived trace payload must be a valid OTLP TracesData protobuf."
        ) from e
    # Archived payloads use a single canonical OTLP wrapper shape. MLflow's trace model is a flat
    # list of spans and does not preserve ResourceSpans / ScopeSpans groupings as first-class data.
    if len(traces_data.resource_spans) != 1:
        raise MlflowException.invalid_parameter_value(
            "Archived trace payload must contain exactly one ResourceSpans group."
        )
    if len(traces_data.resource_spans[0].scope_spans) != 1:
        raise MlflowException.invalid_parameter_value(
            "Archived trace payload must contain exactly one ScopeSpans group."
        )
    spans = [
        Span.from_otel_proto(otel_span)
        for resource_spans in traces_data.resource_spans
        for scope_spans in resource_spans.scope_spans
        for otel_span in scope_spans.spans
    ]
    if not spans:
        raise MlflowException.invalid_parameter_value(
            "Archived trace payload must contain at least one span."
        )
    otlp_trace_ids = {span._trace_id for span in spans}
    if len(otlp_trace_ids) > 1:
        raise MlflowException.invalid_parameter_value(
            "Archived trace payload must contain spans for a single OTLP trace, "
            f"but got distinct trace IDs: {sorted(otlp_trace_ids)}"
        )
    return _sort_spans_for_trace_output(spans)
