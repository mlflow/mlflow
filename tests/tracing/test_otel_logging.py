"""
Tests for OpenTelemetry client integration with MLflow otel_api.py endpoint.

This test suite verifies that the experiment ID header functionality works correctly
when using OpenTelemetry clients to send spans to MLflow's OTel endpoint.
"""

import shutil
import time
from pathlib import Path
from typing import Iterator

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
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.util._once import Once

import mlflow
from mlflow.server import handlers
from mlflow.server.fastapi_app import app as mlflow_app
from mlflow.server.handlers import initialize_backend_stores
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.utils import encode_trace_id
from mlflow.tracing.utils.otlp import MLFLOW_EXPERIMENT_ID_HEADER
from mlflow.version import IS_TRACING_SDK_ONLY

from tests.helper_functions import get_safe_port
from tests.store.tracking.test_sqlalchemy_store import ARTIFACT_URI
from tests.tracking.integration_test_utils import ServerThread

if IS_TRACING_SDK_ONLY:
    pytest.skip("OTel endpoint tests require full MLflow server", allow_module_level=True)


@pytest.fixture(scope="module")
def cached_db(tmp_path_factory) -> Path:
    """Creates and caches a SQLite database to avoid repeated migrations for each test run."""
    tmp_path = tmp_path_factory.mktemp("sqlite_db")
    db_path = tmp_path / "mlflow.db"
    db_uri = f"sqlite:///{db_path}"
    store = SqlAlchemyStore(db_uri, ARTIFACT_URI)
    store.engine.dispose()
    return db_path


@pytest.fixture
def mlflow_server(tmp_path: Path, cached_db: Path) -> Iterator[str]:
    # Copy the pre-initialized cached DB into this test's tmp path
    db_path = tmp_path / "mlflow.db"
    shutil.copy(cached_db, db_path)

    backend_store_uri = f"sqlite:///{db_path}"
    artifact_root = tmp_path.as_uri()

    handlers._tracking_store = None
    handlers._model_registry_store = None
    initialize_backend_stores(backend_store_uri, default_artifact_root=artifact_root)

    # Start the FastAPI app in a background thread and yield its URL.
    with ServerThread(mlflow_app, get_safe_port()) as url:
        yield url


def test_otel_client_sends_spans_to_mlflow_database(mlflow_server: str, monkeypatch):
    """
    Test end-to-end: OpenTelemetry client sends spans via experiment ID header to MLflow.

    Note: This test verifies that spans are successfully accepted by the server.
    Without artifact upload, traces won't be retrievable via search_traces.
    """
    # Enable synchronous trace logging to ensure traces are immediately available
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING", "false")

    mlflow.set_tracking_uri(mlflow_server)

    experiment = mlflow.set_experiment("otel-test-experiment")
    experiment_id = experiment.experiment_id

    resource = OTelSDKResource.create({"service.name": "test-service-e2e"})
    tracer_provider = TracerProvider(resource=resource)

    # First, verify the endpoint is reachable
    test_response = requests.get(f"{mlflow_server}/health", timeout=5)
    assert test_response.status_code == 200, (
        f"Server health check failed: {test_response.status_code}"
    )

    exporter = OTLPSpanExporter(
        endpoint=f"{mlflow_server}/v1/traces",
        headers={MLFLOW_EXPERIMENT_ID_HEADER: experiment_id},
        timeout=10,  # Explicit timeout
    )

    # Use SimpleSpanProcessor for immediate span export in tests
    # This ensures spans are sent immediately rather than batched
    span_processor = SimpleSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)

    # Reset the global tracer provider to avoid conflicts with other tests.
    # This is necessary because OpenTelemetry doesn't allow overriding an already-set provider.
    #
    # NOTE: We're using internal APIs here (_TRACER_PROVIDER_SET_ONCE and _TRACER_PROVIDER)
    # because OpenTelemetry doesn't provide a public API to reset the global tracer provider.
    # The library is designed to set the provider once at application startup, which doesn't
    # work well for testing scenarios where different tests need different configurations.
    # This pattern is also used in tests/semantic_kernel/conftest.py for the same reason.
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    otel_trace._TRACER_PROVIDER = None

    # Set the tracer provider
    otel_trace.set_tracer_provider(tracer_provider)
    tracer = otel_trace.get_tracer(__name__)

    with tracer.start_as_current_span("otel-e2e-test-span") as span:
        span.set_attribute("test.e2e.attribute", "e2e-test-value")
        # Capture the OTel trace ID to verify it matches the MLflow trace ID
        otel_trace_id = span.get_span_context().trace_id
        # Verify the span was actually created and has valid context
        assert span.get_span_context().is_valid, "Span context is not valid"
        assert otel_trace_id != 0, "Trace ID is zero"

    # Add a small delay to ensure the server has processed the spans
    time.sleep(0.5)

    # Wait up to 30 seconds for search_traces() to return a trace
    traces = []
    for _ in range(30):
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


def test_otel_endpoint_requires_experiment_id_header(mlflow_server: str):
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


def test_invalid_otel_span_format_returns_400(mlflow_server: str):
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


def test_missing_required_span_fields_returns_422(mlflow_server: str):
    """
    Test that spans that fail MLflow conversion return HTTP 422.
    """
    # Create protobuf request with missing trace_id (this should cause MLflow conversion to fail)
    span = OTelProtoSpan()
    # Don't set trace_id - this should cause from_otel_proto to fail
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

    assert response.status_code == 422


def test_missing_experiment_id_header_returns_422(mlflow_server: str):
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

    assert response.status_code == 422


def test_invalid_content_type_returns_400(mlflow_server: str):
    """
    Test that invalid Content-Type header returns HTTP 400.
    """
    # Create a valid OTLP request
    span = OTelProtoSpan()
    span.trace_id = b"1234567890123456"
    span.span_id = b"12345678"
    span.name = "test-span"
    span.start_time_unix_nano = 1000000000
    span.end_time_unix_nano = 2000000000

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

    # Send request with incorrect Content-Type
    response = requests.post(
        f"{mlflow_server}/v1/traces",
        data=request.SerializeToString(),
        headers={
            "Content-Type": "application/json",  # Wrong content type
            MLFLOW_EXPERIMENT_ID_HEADER: "test-experiment",
        },
        timeout=10,
    )

    assert response.status_code == 400
    assert "Invalid Content-Type" in response.text


def test_empty_resource_spans_returns_400(mlflow_server: str):
    request = ExportTraceServiceRequest()

    response = requests.post(
        f"{mlflow_server}/v1/traces",
        data=request.SerializeToString(),
        headers={
            "Content-Type": "application/x-protobuf",
            MLFLOW_EXPERIMENT_ID_HEADER: "test-experiment",
        },
        timeout=10,
    )

    assert response.status_code == 400
    assert "no spans found" in response.text
