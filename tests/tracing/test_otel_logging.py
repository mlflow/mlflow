"""
Tests for OpenTelemetry client integration with MLflow otel_api.py endpoint.

This test suite verifies that the experiment ID header functionality works correctly
when using OpenTelemetry clients to send spans to MLflow's OTel endpoint.
"""

import time

import pytest
import requests
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import InstrumentationScope
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans
from opentelemetry.proto.trace.v1.trace_pb2 import Span as OTelProtoSpan
from opentelemetry.sdk.resources import Resource as OTelSDKResource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

import mlflow
from mlflow.tracing.utils import encode_trace_id
from mlflow.tracing.utils.otlp import MLFLOW_EXPERIMENT_ID_HEADER
from mlflow.version import IS_TRACING_SDK_ONLY

from tests.tracking.integration_test_utils import _init_server

if IS_TRACING_SDK_ONLY:
    pytest.skip("OTel endpoint tests require full MLflow server", allow_module_level=True)


@pytest.fixture
def mlflow_server(tmp_path):
    """Fixture to provide a running MLflow server with FastAPI that includes OTel routes."""
    backend_store_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_root = tmp_path.as_uri()

    # Use _init_server with FastAPI (which is now the default)
    with _init_server(backend_store_uri, artifact_root) as url:
        yield url


def test_otel_client_sends_spans_to_mlflow_database(mlflow_server):
    """
    Test end-to-end: OpenTelemetry client sends spans via experiment ID header to MLflow.

    Note: This test verifies that spans are successfully accepted by the server.
    Without artifact upload, traces won't be retrievable via search_traces.
    """
    mlflow.set_tracking_uri(mlflow_server)

    experiment = mlflow.set_experiment("otel-test-experiment")
    experiment_id = experiment.experiment_id

    resource = OTelSDKResource.create({"service.name": "test-service-e2e"})
    tracer_provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(
        endpoint=f"{mlflow_server}/v1/traces", headers={MLFLOW_EXPERIMENT_ID_HEADER: experiment_id}
    )

    span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)

    otel_trace.set_tracer_provider(tracer_provider)
    tracer = otel_trace.get_tracer(__name__)

    with tracer.start_as_current_span("otel-e2e-test-span") as span:
        span.set_attribute("test.e2e.attribute", "e2e-test-value")
        # Capture the OTel trace ID to verify it matches the MLflow trace ID
        otel_trace_id = span.get_span_context().trace_id

    flush_success = span_processor.force_flush(10000)
    assert flush_success, "Failed to flush spans to the server"

    # Give the server a moment to process the spans after they've been sent
    time.sleep(1)

    # Retry a few times in case the server needs more time to process
    traces = []
    for _ in range(3):
        traces = mlflow.search_traces(
            experiment_ids=[experiment_id], include_spans=False, return_type="list"
        )
        if traces:
            break
        time.sleep(1)

    assert len(traces) > 0, "No traces found in the database after sending spans"

    # Verify the trace ID matches the expected format based on the OTel span
    expected_trace_id = f"tr-{encode_trace_id(otel_trace_id)}"
    actual_trace_id = traces[0].info.trace_id
    assert actual_trace_id == expected_trace_id, (
        f"Trace ID mismatch: expected {expected_trace_id}, got {actual_trace_id}"
    )


def test_otel_endpoint_requires_experiment_id_header(mlflow_server):
    """
    Test that the OTel endpoint requires experiment ID header.
    """
    # Create protobuf request
    span = OTelProtoSpan()
    span.trace_id = bytes.fromhex("0000000000000002" + "0" * 16)
    span.span_id = bytes.fromhex("00000002" + "0" * 8)
    span.name = "test-span-no-header"

    scope = InstrumentationScope()
    scope.name = "test-scope"

    scope_spans = ScopeSpans()
    scope_spans.scope.CopyFrom(scope)
    scope_spans.spans.append(span)

    resource = Resource()
    resource_spans = ResourceSpans()
    resource_spans.resource.CopyFrom(resource)
    resource_spans.scope_spans.append(scope_spans)

    request = ExportTraceServiceRequest()
    request.resource_spans.append(resource_spans)

    response = requests.post(
        f"{mlflow_server}/v1/traces",
        data=request.SerializeToString(),
        headers={"Content-Type": "application/x-protobuf"},
        timeout=10,
    )

    assert response.status_code == 422


def test_invalid_otel_span_format_returns_400(mlflow_server):
    """
    Test that invalid OpenTelemetry protobuf format returns HTTP 400.
    """
    # Send completely invalid protobuf data
    invalid_protobuf_data = b"this is not valid protobuf data at all"

    response = requests.post(
        f"{mlflow_server}/v1/traces",
        data=invalid_protobuf_data,
        headers={
            "Content-Type": "application/x-protobuf",
            MLFLOW_EXPERIMENT_ID_HEADER: "test-experiment",
        },
        timeout=10,
    )

    assert response.status_code == 400, f"Expected 400, got {response.status_code}"


def test_missing_required_span_fields_returns_422(mlflow_server):
    """
    Test that spans that fail MLflow conversion return HTTP 422.
    """
    # Create protobuf request with missing trace_id (this should cause MLflow conversion to fail)
    span = OTelProtoSpan()
    # Don't set trace_id - this should cause _from_otel_proto to fail
    span.span_id = bytes.fromhex("00000001" + "0" * 8)
    span.name = "incomplete-span"

    scope = InstrumentationScope()
    scope.name = "test-scope"

    scope_spans = ScopeSpans()
    scope_spans.scope.CopyFrom(scope)
    scope_spans.spans.append(span)

    resource = Resource()
    resource_spans = ResourceSpans()
    resource_spans.resource.CopyFrom(resource)
    resource_spans.scope_spans.append(scope_spans)

    request = ExportTraceServiceRequest()
    request.resource_spans.append(resource_spans)

    response = requests.post(
        f"{mlflow_server}/v1/traces",
        data=request.SerializeToString(),
        headers={
            "Content-Type": "application/x-protobuf",
            MLFLOW_EXPERIMENT_ID_HEADER: "test-experiment",
        },
        timeout=10,
    )

    assert response.status_code == 422, f"Expected 422, got {response.status_code}"


def test_missing_experiment_id_header_returns_422(mlflow_server):
    """
    Test that missing experiment ID header returns HTTP 422 (FastAPI validation error).
    """
    # Create valid protobuf request
    span = OTelProtoSpan()
    span.trace_id = bytes.fromhex("0000000000000003" + "0" * 16)
    span.span_id = bytes.fromhex("00000003" + "0" * 8)
    span.name = "test-span"

    scope = InstrumentationScope()
    scope.name = "test-scope"

    scope_spans = ScopeSpans()
    scope_spans.scope.CopyFrom(scope)
    scope_spans.spans.append(span)

    resource = Resource()
    resource_spans = ResourceSpans()
    resource_spans.resource.CopyFrom(resource)
    resource_spans.scope_spans.append(scope_spans)

    request = ExportTraceServiceRequest()
    request.resource_spans.append(resource_spans)

    response = requests.post(
        f"{mlflow_server}/v1/traces",
        data=request.SerializeToString(),
        headers={"Content-Type": "application/x-protobuf"},
        timeout=10,
    )

    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
