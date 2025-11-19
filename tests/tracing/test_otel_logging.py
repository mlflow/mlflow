"""
Tests for OpenTelemetry client integration with MLflow otel_api.py endpoint.

This test suite verifies that the experiment ID header functionality works correctly
when using OpenTelemetry clients to send spans to MLflow's OTel endpoint.
"""

import shutil
import time
from pathlib import Path
from typing import Iterator
from unittest import mock

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
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.util._once import Once

import mlflow
from mlflow.server import handlers
from mlflow.server.fastapi_app import app as mlflow_app
from mlflow.server.handlers import initialize_backend_stores
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.telemetry.client import TelemetryClient
from mlflow.telemetry.events import TraceSource, TracesReceivedByServerEvent
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


def test_batch_span_processor_with_multiple_traces(mlflow_server: str):
    """
    Test that BatchSpanProcessor can send spans from multiple traces in a single request.
    This verifies the server-side grouping by trace_id functionality.
    """
    mlflow.set_tracking_uri(mlflow_server)

    experiment = mlflow.set_experiment("otel-batch-test-experiment")
    experiment_id = experiment.experiment_id

    resource = OTelSDKResource.create({"service.name": "test-batch-service"})
    tracer_provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(
        endpoint=f"{mlflow_server}/v1/traces",
        headers={MLFLOW_EXPERIMENT_ID_HEADER: experiment_id},
        timeout=10,
    )

    # Use BatchSpanProcessor to batch spans from multiple traces
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)

    # Reset the global tracer provider
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    otel_trace._TRACER_PROVIDER = None
    otel_trace.set_tracer_provider(tracer_provider)

    tracer = otel_trace.get_tracer(__name__)

    # Create multiple traces with spans
    trace_ids = []
    for i in range(3):
        with tracer.start_as_current_span(f"batch-test-span-{i}") as span:
            span.set_attribute("test.batch.index", i)
            otel_trace_id = span.get_span_context().trace_id
            trace_ids.append(otel_trace_id)
            assert otel_trace_id != 0

    # Force flush to send all batched spans
    span_processor.force_flush()

    traces = mlflow.search_traces(
        experiment_ids=[experiment_id], include_spans=False, return_type="list"
    )

    assert len(traces) == 3

    # Verify all expected trace IDs are present
    expected_trace_ids = {f"tr-{encode_trace_id(tid)}" for tid in trace_ids}
    actual_trace_ids = {trace.info.trace_id for trace in traces}

    assert expected_trace_ids == actual_trace_ids


def test_multiple_traces_in_single_request(mlflow_server: str):
    """
    Test that a single request containing spans from multiple traces is handled correctly.
    This simulates what BatchSpanProcessor does internally.
    """

    mlflow.set_tracking_uri(mlflow_server)
    experiment = mlflow.set_experiment("otel-multi-trace-test")
    experiment_id = experiment.experiment_id

    # Create protobuf request with spans from 3 different traces
    request = ExportTraceServiceRequest()

    for trace_num in range(3):
        # Create a span with unique trace_id
        span = OTelProtoSpan()
        trace_id_hex = f"{trace_num:016x}" + "0" * 16
        span.trace_id = bytes.fromhex(trace_id_hex)
        span.span_id = bytes.fromhex(f"{trace_num:08x}" + "0" * 8)
        span.name = f"multi-trace-span-{trace_num}"
        span.start_time_unix_nano = 1000000000 + trace_num * 1000
        span.end_time_unix_nano = 2000000000 + trace_num * 1000

        scope = InstrumentationScope()
        scope.name = "test-scope"

        scope_spans = ScopeSpans()
        scope_spans.scope.CopyFrom(scope)
        scope_spans.spans.append(span)

        resource = Resource()
        resource_spans = ResourceSpans()
        resource_spans.resource.CopyFrom(resource)
        resource_spans.scope_spans.append(scope_spans)

        request.resource_spans.append(resource_spans)

    # Send the request with multiple traces
    response = requests.post(
        f"{mlflow_server}/v1/traces",
        data=request.SerializeToString(),
        headers={
            "Content-Type": "application/x-protobuf",
            MLFLOW_EXPERIMENT_ID_HEADER: experiment_id,
        },
        timeout=10,
    )

    assert response.status_code == 200

    traces = mlflow.search_traces(
        experiment_ids=[experiment_id], include_spans=False, return_type="list"
    )

    assert len(traces) == 3


def test_logging_many_traces_in_single_request(mlflow_server: str):
    mlflow.set_tracking_uri(mlflow_server)
    experiment = mlflow.set_experiment("otel-many-traces-test")
    experiment_id = experiment.experiment_id

    # Create a request with 15 different traces (exceeds the 10 thread pool limit)
    request = ExportTraceServiceRequest()
    num_traces = 15

    for trace_num in range(num_traces):
        span = OTelProtoSpan()
        trace_id_hex = f"{trace_num + 1000:016x}" + "0" * 16
        span.trace_id = bytes.fromhex(trace_id_hex)
        span.span_id = bytes.fromhex(f"{trace_num + 1000:08x}" + "0" * 8)
        span.name = f"many-traces-test-span-{trace_num}"
        span.start_time_unix_nano = 1000000000 + trace_num * 1000
        span.end_time_unix_nano = 2000000000 + trace_num * 1000

        scope = InstrumentationScope()
        scope.name = "many-traces-test-scope"

        scope_spans = ScopeSpans()
        scope_spans.scope.CopyFrom(scope)
        scope_spans.spans.append(span)

        resource = Resource()
        resource_spans = ResourceSpans()
        resource_spans.resource.CopyFrom(resource)
        resource_spans.scope_spans.append(scope_spans)

        request.resource_spans.append(resource_spans)

    # Send the request and measure response time
    requests.post(
        f"{mlflow_server}/v1/traces",
        data=request.SerializeToString(),
        headers={
            "Content-Type": "application/x-protobuf",
            MLFLOW_EXPERIMENT_ID_HEADER: experiment_id,
        },
        timeout=10,
    )

    traces = mlflow.search_traces(
        experiment_ids=[experiment_id], include_spans=False, return_type="list"
    )

    assert len(traces) == num_traces


def test_mixed_trace_spans_in_single_request(mlflow_server: str):
    """
    Test that multiple spans from the same trace, mixed with spans from other traces,
    are grouped and logged correctly.
    """
    mlflow.set_tracking_uri(mlflow_server)
    experiment = mlflow.set_experiment("otel-mixed-test")
    experiment_id = experiment.experiment_id

    request = ExportTraceServiceRequest()

    # Create 2 spans for trace A, 1 span for trace B, 2 spans for trace C
    trace_configs = [
        ("A", 0, 2),  # trace A with 2 spans
        ("B", 1, 1),  # trace B with 1 span
        ("C", 2, 2),  # trace C with 2 spans
    ]

    for trace_name, trace_id_num, span_count in trace_configs:
        trace_id_hex = f"{trace_id_num + 2000:016x}" + "0" * 16

        for span_num in range(span_count):
            span = OTelProtoSpan(
                trace_id=bytes.fromhex(trace_id_hex),
                span_id=bytes.fromhex(f"{trace_id_num * 100 + span_num:08x}" + "0" * 8),
                name=f"mixed-span-{trace_name}-{span_num}",
                start_time_unix_nano=1000000000 + span_num * 1000,
                end_time_unix_nano=2000000000 + span_num * 1000,
            )

            scope = InstrumentationScope(
                name="mixed-test-scope",
            )

            scope_spans = ScopeSpans()
            scope_spans.scope.CopyFrom(scope)
            scope_spans.spans.append(span)

            resource = Resource()
            resource_spans = ResourceSpans()
            resource_spans.resource.CopyFrom(resource)
            resource_spans.scope_spans.append(scope_spans)

            request.resource_spans.append(resource_spans)

    response = requests.post(
        f"{mlflow_server}/v1/traces",
        data=request.SerializeToString(),
        headers={
            "Content-Type": "application/x-protobuf",
            MLFLOW_EXPERIMENT_ID_HEADER: experiment_id,
        },
        timeout=10,
    )

    assert response.status_code == 200

    traces = mlflow.search_traces(
        experiment_ids=[experiment_id], include_spans=True, return_type="list"
    )

    assert len(traces) == 3
    span_counts = [len(trace.data.spans) for trace in traces]
    assert span_counts == [2, 1, 2]


def test_error_logging_spans(mlflow_server: str):
    mlflow.set_tracking_uri(mlflow_server)
    experiment = mlflow.set_experiment("otel-error-test")
    experiment_id = experiment.experiment_id

    resource = OTelSDKResource.create({"service.name": "test-batch-service"})
    tracer_provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(
        endpoint=f"{mlflow_server}/v1/traces",
        headers={MLFLOW_EXPERIMENT_ID_HEADER: experiment_id},
        timeout=10,
    )

    # Use BatchSpanProcessor to batch spans from multiple traces
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)

    # Reset the global tracer provider
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    otel_trace._TRACER_PROVIDER = None
    otel_trace.set_tracer_provider(tracer_provider)

    tracer = otel_trace.get_tracer(__name__)

    original_log_spans = SqlAlchemyStore.log_spans
    call_count = {"count": 0}

    def mock_log_spans(self, *args, **kwargs):
        if call_count["count"] == 0:
            call_count["count"] += 1
            raise Exception("test_error")
        else:
            return original_log_spans(self, *args, **kwargs)

    with (
        mock.patch.object(SqlAlchemyStore, "log_spans", mock_log_spans),
        mock.patch(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter._logger.error"
        ) as mock_error,
    ):
        for _ in range(2):
            with tracer.start_as_current_span("batch-test-span-0"):
                pass

        span_processor.force_flush()

        assert any(
            "Failed to log OpenTelemetry spans" in error[0][2]
            for error in mock_error.call_args_list
        )
        assert any("test_error" in error[0][2] for error in mock_error.call_args_list)

    traces = mlflow.search_traces(
        experiment_ids=[experiment_id], include_spans=False, return_type="list"
    )

    assert len(traces) == 1


def test_otel_trace_received_telemetry_from_mlflow_client(mlflow_server: str):
    """
    Test TraceReceivedByServerEvent telemetry shows source=MLFLOW_PYTHON_CLIENT for standard client.

    Uses @mlflow.trace with standard MLflow client configuration, which automatically sends
    User-Agent and X-MLflow-Client-Version headers to identify traces from MLflow client.
    """
    mlflow.set_tracking_uri(mlflow_server)
    mlflow.set_experiment("otel-telemetry-mlflow-client-test")

    with mock.patch("mlflow.telemetry.track.get_telemetry_client") as mock_get_client:
        mock_client = mock.MagicMock(spec=TelemetryClient)
        mock_get_client.return_value = mock_client

        @mlflow.trace
        def test_function():
            return "test result"

        result = test_function()
        assert result == "test result"

        time.sleep(1)

        if mock_client.add_record.called:
            record = mock_client.add_record.call_args[0][0]
            assert record.event_name == TracesReceivedByServerEvent.name
            assert record.params["source"] == TraceSource.MLFLOW_PYTHON_CLIENT.value
            assert record.params["count"] == 1


def test_otel_trace_received_telemetry_from_external_client(mlflow_server: str):
    """
    Test TracesReceivedByServerEvent telemetry shows source=UNKNOWN for external clients.

    Sends a direct protobuf request without MLflow client headers to simulate an external
    OpenTelemetry client (not MLflow client). Tests with 2 traces to verify count field.
    """
    mlflow.set_tracking_uri(mlflow_server)
    experiment = mlflow.set_experiment("otel-telemetry-external-client-test")
    experiment_id = experiment.experiment_id

    trace_id_1 = bytes.fromhex("0000000000000100" + "0" * 16)
    trace_id_2 = bytes.fromhex("0000000000000200" + "0" * 16)

    request = ExportTraceServiceRequest()

    # First trace with root span and child spans
    request.resource_spans.append(
        ResourceSpans(
            scope_spans=[
                ScopeSpans(
                    scope=InstrumentationScope(name="telemetry-test-scope"),
                    spans=[
                        OTelProtoSpan(
                            trace_id=trace_id_1,
                            span_id=bytes.fromhex("00000001" + "0" * 8),
                            name="root-span-1",
                            start_time_unix_nano=1000000000,
                            end_time_unix_nano=2000000000,
                        ),
                        OTelProtoSpan(
                            trace_id=trace_id_1,
                            span_id=bytes.fromhex("00000002" + "0" * 8),
                            parent_span_id=bytes.fromhex("00000001" + "0" * 8),
                            name="child-span-1",
                            start_time_unix_nano=1100000000,
                            end_time_unix_nano=1500000000,
                        ),
                    ],
                )
            ]
        )
    )

    # Second trace with root span
    request.resource_spans.append(
        ResourceSpans(
            scope_spans=[
                ScopeSpans(
                    scope=InstrumentationScope(name="telemetry-test-scope"),
                    spans=[
                        OTelProtoSpan(
                            trace_id=trace_id_2,
                            span_id=bytes.fromhex("00000003" + "0" * 8),
                            name="root-span-2",
                            start_time_unix_nano=1600000000,
                            end_time_unix_nano=1900000000,
                        ),
                    ],
                )
            ]
        )
    )

    with mock.patch("mlflow.telemetry.track.get_telemetry_client") as mock_get_client:
        mock_client = mock.MagicMock(spec=TelemetryClient)
        mock_get_client.return_value = mock_client

        response = requests.post(
            f"{mlflow_server}/v1/traces",
            data=request.SerializeToString(),
            headers={
                "Content-Type": "application/x-protobuf",
                MLFLOW_EXPERIMENT_ID_HEADER: experiment_id,
            },
            timeout=10,
        )

        assert response.status_code == 200

        mock_client.add_record.assert_called_once()
        record = mock_client.add_record.call_args[0][0]

        assert record.event_name == TracesReceivedByServerEvent.name
        assert record.status.value == "success"
        assert record.params["source"] == TraceSource.UNKNOWN.value
        assert record.params["count"] == 2
