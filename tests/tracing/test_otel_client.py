"""
Tests for OpenTelemetry client integration with MLflow otel_api.py endpoint.

This test suite verifies that the experiment ID header functionality works correctly
when using OpenTelemetry clients to send spans to MLflow's OTel endpoint.
"""

import subprocess
import sys

import pytest
import requests
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
from mlflow.tracing.utils.otlp import MLFLOW_EXPERIMENT_ID_HEADER

from tests.helper_functions import get_safe_port
from tests.tracking.integration_test_utils import _await_server_up_or_die


@pytest.fixture
def mlflow_server(tmp_path):
    """Fixture to provide a running MLflow server with FastAPI that includes OTel routes."""
    mlflow.set_tracking_uri(None)
    backend_store_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_root = tmp_path.as_uri()
    server_port = get_safe_port()

    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            backend_store_uri,
            "--default-artifact-root",
            artifact_root,
            "--host",
            "127.0.0.1",
            "--port",
            str(server_port),
        ],
    ) as proc:
        try:
            _await_server_up_or_die(server_port)
            yield f"http://127.0.0.1:{server_port}"
        finally:
            proc.terminate()


def test_otel_client_sends_spans_to_mlflow_database(mlflow_server):
    """
    Test end-to-end: OpenTelemetry client sends spans via experiment ID header to MLflow.

    Note: This test verifies that spans are successfully accepted by the server.
    Without artifact upload, traces won't be retrievable via search_traces.
    """
    import mlflow

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

    from opentelemetry import trace as otel_trace

    otel_trace.set_tracer_provider(tracer_provider)
    tracer = otel_trace.get_tracer(__name__)

    with tracer.start_as_current_span("otel-e2e-test-span") as span:
        span.set_attribute("test.e2e.attribute", "e2e-test-value")

    # Force flush returns True if successful
    flush_success = span_processor.force_flush(10000)  # 10 second timeout

    # The test passes if the spans were successfully sent (no exceptions raised)
    # and the processor successfully flushed
    assert flush_success, "Failed to flush spans to the server"


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
