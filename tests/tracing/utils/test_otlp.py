import gzip
import time
import zlib
from collections.abc import Callable

import pytest
from fastapi import HTTPException
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor

import mlflow
from mlflow.entities.span import SpanType
from mlflow.environment_variables import MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.processor.mlflow_v3 import MlflowV3SpanProcessor
from mlflow.tracing.processor.otel import OtelSpanProcessor
from mlflow.tracing.provider import _get_trace_exporter, _get_tracer
from mlflow.tracking import MlflowClient
from mlflow.utils.os import is_windows

from tests.tracing.helper import get_traces

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
from mlflow.tracing.utils.otlp import (
    decompress_otlp_body,
    get_otlp_exporter,
    should_use_otlp_exporter,
)

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
@pytest.mark.parametrize("dual_export", [True, False, None], ids=["enable", "disable", "default"])
def test_export_to_otel_collector(otel_collector, monkeypatch, dual_export):
    if dual_export:
        monkeypatch.setenv("MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT", "true")
    elif dual_export is False:
        monkeypatch.setenv("MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT", "false")

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

    model = TestModel()
    model.predict(2, 5)

    # Tracer should be configured to export to OTLP
    exporter = _get_trace_exporter()
    assert isinstance(exporter, OTLPSpanExporter)
    assert exporter._endpoint == f"127.0.0.1:{port}"

    mlflow_traces = get_traces()
    if dual_export:
        assert len(mlflow_traces) == 1
        assert mlflow_traces[0].info.state == "OK"
        assert len(mlflow_traces[0].data.spans) == 3
    else:
        assert len(mlflow_traces) == 0

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
        if (
            "predict" in collector_logs
            and "add_one_with_custom_name" in collector_logs
            and "square" in collector_logs
        ):
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
    traces = client.search_traces(locations=[experiment.experiment_id])
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


@pytest.mark.parametrize(
    ("encoding", "compress_fn", "data"),
    [
        ("gzip", gzip.compress, b"otlp-data-test"),
        ("deflate", zlib.compress, b"otlp-deflate-data"),
        ("deflate", lambda d: zlib.compress(d)[2:-4], b"raw-deflate-data"),  # Raw deflate
    ],
    ids=["gzip", "deflate-rfc", "deflate-raw"],
)
def test_decompress_otlp_body_valid(
    encoding: str, compress_fn: Callable[[bytes], bytes], data: bytes
):
    compressed = compress_fn(data)
    output = decompress_otlp_body(compressed, encoding)
    assert output == data


@pytest.mark.parametrize(
    ("encoding", "invalid_data", "expected_error"),
    [
        ("gzip", b"not-gzip-data", r"Failed to decompress gzip payload"),
        ("deflate", b"not-deflate-data", r"Failed to decompress deflate payload"),
        ("unknown-encoding", b"xxx", r"Unsupported Content-Encoding"),
    ],
    ids=["gzip-invalid", "deflate-invalid", "unknown-encoding"],
)
def test_decompress_otlp_body_invalid(encoding: str, invalid_data: bytes, expected_error: str):
    with pytest.raises(HTTPException, match=expected_error, check=lambda e: e.status_code == 400):
        decompress_otlp_body(invalid_data, encoding)


def test_metadata_added_in_root_span_with_otel_export(monkeypatch):
    saved_spans = []

    def mock_on_end(self, span: OTelReadableSpan):
        saved_spans.append(span)

    # Endpoint not used as on_end is mocked
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:42/v1/traces")
    monkeypatch.setattr(BatchSpanProcessor, "on_end", mock_on_end)
    mlflow.set_experiment("metadata_export_test")

    processors = _get_tracer("test").span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], OtelSpanProcessor)

    @mlflow.trace(name="parent_span")
    def parent_function():
        result = child_function("Hello", "World")
        return f"Parent: {result}"

    @mlflow.trace(name="child_span")
    def child_function(arg1, arg2):
        mlflow.update_current_trace(
            metadata={"str": "42", "int": 123, "obj": {"hello": "world"}},
        )
        return f"{arg1} {arg2}"

    result = parent_function()
    assert result == "Parent: Hello World"

    assert len(saved_spans) == 2

    for span in saved_spans:
        if span.parent is None:
            assert span.attributes.get(SpanAttributeKey.METADATA.format(key="str")) == '"42"'
            assert span.attributes.get(SpanAttributeKey.METADATA.format(key="int")) == "123"
            assert (
                span.attributes.get(SpanAttributeKey.METADATA.format(key="obj"))
                == '{"hello": "world"}'
            )
            assert any(
                k.startswith(SpanAttributeKey.METADATA.format(key="mlflow"))
                for k in span.attributes.keys()
            )
        else:
            assert span.attributes.get(SpanAttributeKey.METADATA.format(key="str")) is None
            assert span.attributes.get(SpanAttributeKey.METADATA.format(key="int")) is None
            assert span.attributes.get(SpanAttributeKey.METADATA.format(key="obj")) is None
            assert not any(
                k.startswith(SpanAttributeKey.METADATA.format(key="mlflow"))
                for k in span.attributes.keys()
            )

    # Test exception when setting metadata
    def mock_get_mlflow_span(span):
        raise Exception("Simulated error during metadata retrieval")

    monkeypatch.setattr(
        "mlflow.tracing.processor.otel.get_mlflow_span_for_otel_span", mock_get_mlflow_span
    )
    saved_spans = []
    result = parent_function()
    assert result == "Parent: Hello World"
    assert len(saved_spans) == 2
