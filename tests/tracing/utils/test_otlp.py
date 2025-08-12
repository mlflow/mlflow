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


def test_should_use_otlp_exporter(monkeypatch):
    assert not should_use_otlp_exporter()

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", _TEST_HTTP_OTLP_ENDPOINT)
    assert should_use_otlp_exporter()

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _TEST_HTTP_OTLP_ENDPOINT)
    assert should_use_otlp_exporter()


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

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:4317/v1/traces")

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
        time.sleep(10)

    # Tracer should be configured to export to OTLP
    exporter = _get_trace_exporter()
    assert isinstance(exporter, OTLPSpanExporter)
    assert exporter._endpoint == "127.0.0.1:4317"

    # Traces should not be logged to MLflow
    mock_client.start_trace.assert_not_called()
    mock_client._upload_trace_data.assert_not_called()
    mock_client._upload_ended_trace_info.assert_not_called()

    # Analyze the logs of the collector
    _, output_file = otel_collector
    with open(output_file) as f:
        collector_logs = f.read()

    # 3 spans should be exported
    assert "Span #0" in collector_logs
    assert "Span #1" in collector_logs
    assert "Span #2" in collector_logs
    assert "Span #3" not in collector_logs


@pytest.mark.skipif(is_windows(), reason="Otel collector docker image does not support Windows")
def test_dual_export_to_mlflow_and_otel(otel_collector, monkeypatch):
    """
    Test that dual export mode sends traces to both MLflow and OTLP collector.
    """
    monkeypatch.setenv(MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT.name, "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:4317/v1/traces")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")

    experiment = mlflow.set_experiment("dual_export_test")
    mlflow.tracing.reset()

    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors
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

    # Wait for traces to be exported to OTLP
    time.sleep(5)

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

    # Verify same trace/span IDs in both backends
    mlflow_span_ids = [span.span_id for span in trace.data.spans]
    trace_id = trace.info.trace_id.replace("tr-", "")
    _, output_file = otel_collector
    with open(output_file) as f:
        collector_logs = f.read()
    assert trace_id in collector_logs
    assert "Span #0" in collector_logs
    assert "Span #1" in collector_logs
    for span_id in mlflow_span_ids:
        assert span_id in collector_logs
