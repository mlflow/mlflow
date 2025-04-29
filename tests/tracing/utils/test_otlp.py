import time

import pytest

import mlflow
from mlflow.entities.span import SpanType
from mlflow.tracing.provider import _get_trace_exporter
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
def test_export_to_otel_collector(otel_collector, mock_client, monkeypatch):
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
