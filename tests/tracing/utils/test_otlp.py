import time
from unittest import mock

import pytest

import mlflow
from mlflow.entities.span import SpanType
from mlflow.environment_variables import MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT
from mlflow.tracing.processor.mlflow_v3 import MlflowV3SpanProcessor
from mlflow.tracing.processor.otel import OtelSpanProcessor
from mlflow.tracing.provider import _get_trace_exporter, _get_tracer
from mlflow.tracking import MlflowClient
from mlflow.utils.os import is_windows

# OTLP exporters are not installed in some CI jobs
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GrpcExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HttpExporter,
    )
except ImportError:
    pytest.skip("OTLP exporters are not installed", allow_module_level=True)

from mlflow.exceptions import MlflowException
from mlflow.tracing.utils.otlp import get_otlp_exporter, should_use_otlp_exporter

_TEST_HTTP_OTLP_ENDPOINT = "http://127.0.0.1:4317/v1/traces"
_TEST_HTTPS_OTLP_ENDPOINT = "https://127.0.0.1:4317/v1/traces"


@pytest.mark.parametrize(
    ("traces_endpoint", "otlp_endpoint", "mlflow_enable", "expected"),
    [
        # No endpoints configured
        (None, None, None, False),  # Default behavior - no export without endpoint
        (None, None, "true", False),  # Explicit enable but no endpoint
        (None, None, "false", False),  # Explicit disable and no endpoint
        # OTEL_EXPORTER_OTLP_TRACES_ENDPOINT configured
        (_TEST_HTTP_OTLP_ENDPOINT, None, None, True),  # Default behavior - export enabled
        (_TEST_HTTP_OTLP_ENDPOINT, None, "true", True),  # Explicit enable
        (_TEST_HTTP_OTLP_ENDPOINT, None, "false", False),  # Explicit disable
        # OTEL_EXPORTER_OTLP_ENDPOINT configured
        (None, _TEST_HTTP_OTLP_ENDPOINT, None, True),  # Default behavior - export enabled
        (None, _TEST_HTTP_OTLP_ENDPOINT, "true", True),  # Explicit enable
        (None, _TEST_HTTP_OTLP_ENDPOINT, "false", False),  # Explicit disable
        # Both endpoints configured (traces endpoint takes precedence)
        (_TEST_HTTP_OTLP_ENDPOINT, _TEST_HTTPS_OTLP_ENDPOINT, None, True),
        (_TEST_HTTP_OTLP_ENDPOINT, _TEST_HTTPS_OTLP_ENDPOINT, "false", False),
    ],
)
def test_should_use_otlp_exporter(
    traces_endpoint, otlp_endpoint, mlflow_enable, expected, monkeypatch
):
    # Clear all relevant environment variables to ensure test isolation
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("MLFLOW_ENABLE_OTLP_EXPORTER", raising=False)

    # Set environment variables based on test parameters
    if traces_endpoint is not None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", traces_endpoint)
    if otlp_endpoint is not None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", otlp_endpoint)
    if mlflow_enable is not None:
        monkeypatch.setenv("MLFLOW_ENABLE_OTLP_EXPORTER", mlflow_enable)

    assert should_use_otlp_exporter() is expected


@pytest.mark.parametrize(
    ("endpoint", "protocol", "expected_type"),
    [
        (_TEST_HTTP_OTLP_ENDPOINT, None, GrpcExporter),
        (_TEST_HTTP_OTLP_ENDPOINT, "grpc", GrpcExporter),
        (_TEST_HTTPS_OTLP_ENDPOINT, "grpc", GrpcExporter),
        (_TEST_HTTP_OTLP_ENDPOINT, "http/protobuf", HttpExporter),
        (_TEST_HTTPS_OTLP_ENDPOINT, "http/protobuf", HttpExporter),
    ],
)
def test_get_otlp_exporter_success(endpoint, protocol, expected_type, monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", endpoint)
    if protocol:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", protocol)

    exporter = get_otlp_exporter()
    assert isinstance(exporter, expected_type)

    if isinstance(exporter, GrpcExporter):
        assert exporter._endpoint == "127.0.0.1:4317"
    else:
        assert exporter._endpoint == endpoint


def test_get_otlp_exporter_invalid_protocol(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", _TEST_HTTP_OTLP_ENDPOINT)

    with pytest.raises(MlflowException, match="Unsupported OTLP protocol"):
        get_otlp_exporter()


@pytest.mark.skipif(is_windows(), reason="Otel collector docker image does not support Windows")
def test_export_to_otel_collector(otel_collector, monkeypatch):
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    _, _, port = otel_collector
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"http://127.0.0.1:{port}/v1/traces")

    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            z = x + y
            z = self.add_one(z)
            z = mlflow.trace(self.square)(z)
            return z  # noqa: RET504

        @mlflow.trace(
            span_type=SpanType.LLM, name="add_one_with_custom_name", attributes={"delta": 1}
        )
        def add_one(self, z):
            return z + 1

        def square(self, t):
            res = t**2
            time.sleep(0.1)
            return res

    mock_client = mock.MagicMock()
    with mock.patch("mlflow.tracing.fluent.TracingClient", return_value=mock_client):
        # Create a trace
        model = TestModel()
        model.predict(2, 5)

    # Tracer should be configured to export to OTLP
    exporter = _get_trace_exporter()
    assert isinstance(exporter, OTLPSpanExporter)
    assert exporter._endpoint == f"127.0.0.1:{port}"

    # Traces should not be logged to MLflow
    mock_client.start_trace.assert_not_called()
    mock_client._upload_trace_data.assert_not_called()
    mock_client._upload_ended_trace_info.assert_not_called()

    # Wait for collector to receive spans, checking every second for up to 60 seconds
    _, output_file, _ = otel_collector
    spans_found = False
    for _ in range(60):
        time.sleep(1)
        with open(output_file) as f:
            collector_logs = f.read()
        # Check if spans are in the logs - the debug exporter outputs span details
        # The BatchSpanProcessor may send spans in multiple batches, so we check for any evidence
        # that the collector is receiving spans from our test
        if "predict" in collector_logs:
            # We found evidence that spans are being exported to the collector
            # The child spans may come in separate batches, but OTLP export works
            spans_found = True
            break

    # Assert that spans were found in collector logs
    assert spans_found, (
        f"Expected spans not found in collector logs after 60 seconds. "
        f"Logs: {collector_logs[:2000]}"
    )


@pytest.mark.skipif(is_windows(), reason="Otel collector docker image does not support Windows")
def test_dual_export_to_mlflow_and_otel(otel_collector, monkeypatch):
    """
    Test that dual export mode sends traces to both MLflow and OTLP collector.
    """
    _, _, port = otel_collector
    monkeypatch.setenv(MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT.name, "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"http://127.0.0.1:{port}/v1/traces")

    experiment = mlflow.set_experiment("dual_export_test")

    processors = _get_tracer("test").span_processor._span_processors
    assert len(processors) == 2
    assert isinstance(processors[0], OtelSpanProcessor)
    assert isinstance(processors[1], MlflowV3SpanProcessor)

    @mlflow.trace(name="parent_span")
    def parent_function():
        result = child_function("Hello", "World")
        return f"Parent: {result}"

    @mlflow.trace(name="child_span")
    def child_function(arg1, arg2):
        # Test that update_current_trace works in dual export mode
        mlflow.update_current_trace({"env": "production", "version": "1.0"})
        return f"{arg1} {arg2}"

    result = parent_function()
    assert result == "Parent: Hello World"

    client = MlflowClient()
    traces = client.search_traces(experiment_ids=[experiment.experiment_id])
    assert len(traces) == 1
    trace = traces[0]
    assert len(trace.data.spans) == 2

    # Verify trace tags were set correctly
    assert "env" in trace.info.tags
    assert trace.info.tags["env"] == "production"
    assert "version" in trace.info.tags
    assert trace.info.tags["version"] == "1.0"

    # Wait for collector to receive spans, checking every second for up to 60 seconds
    _, output_file, _ = otel_collector
    spans_found = False
    for _ in range(60):
        time.sleep(1)
        with open(output_file) as f:
            collector_logs = f.read()
        # Check if spans are in the logs - the debug exporter outputs span details
        # Look for evidence that spans were received
        if "parent_span" in collector_logs or "child_span" in collector_logs:
            # Evidence of traces being exported to OTLP
            spans_found = True
            break

    # Assert that spans were found in collector logs
    assert spans_found, (
        f"Expected spans not found in collector logs after 60 seconds. "
        f"Logs: {collector_logs[:2000]}"
    )
