import time

import pytest
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

import mlflow


@pytest.fixture
def metric_reader() -> InMemoryMetricReader:
    """Create an in-memory metric reader for testing."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    yield reader
    provider.shutdown()


def test_metrics_export(
    monkeypatch: pytest.MonkeyPatch, metric_reader: InMemoryMetricReader
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://localhost:9090")
    mlflow.set_experiment("test_experiment")

    @mlflow.trace(span_type="CHAIN", name="parent")
    def parent_function() -> str:
        mlflow.update_current_trace({"env": "test", "version": "1.0"})
        time.sleep(0.01)  # 10ms
        return child_function()

    @mlflow.trace(span_type="LLM", name="child")
    def child_function() -> str:
        time.sleep(0.25)  # 250ms
        return "result"

    @mlflow.trace(span_type="TOOL", name="error_function")
    def error_function() -> None:
        time.sleep(1.0)  # 1000ms
        raise ValueError("Test error")

    # Execute successful trace
    parent_function()
    # Execute error trace
    with pytest.raises(ValueError, match="Test error"):
        error_function()

    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None

    data_points = []
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == "mlflow.trace.span.duration":
                    assert metric.unit == "ms"
                    data_points.extend(metric.data.data_points)

    assert len(data_points) == 3
    data_points.sort(key=lambda dp: dp.attributes["span_type"])
    chain_metric, llm_metric, tool_metric = data_points

    # LLM span (child) - 250ms
    llm_metric_attrs = dict(llm_metric.attributes)
    assert llm_metric_attrs["span_type"] == "LLM", data_points
    assert llm_metric_attrs["span_status"] == "OK"
    assert llm_metric_attrs["root"] is False
    assert llm_metric.sum >= 250

    # CHAIN span (parent) - includes child time, so ~260ms total
    chain_metric_attrs = dict(chain_metric.attributes)
    assert chain_metric_attrs["span_type"] == "CHAIN", data_points
    assert chain_metric_attrs["span_status"] == "OK"
    assert chain_metric_attrs["root"] is True
    assert chain_metric_attrs["tags.env"] == "test"
    assert chain_metric_attrs["tags.version"] == "1.0"
    assert chain_metric.sum >= 260

    # TOOL span (error) - 1000ms
    tool_metric_attrs = dict(tool_metric.attributes)
    assert tool_metric_attrs["span_type"] == "TOOL", data_points
    assert tool_metric_attrs["span_status"] == "ERROR"
    assert tool_metric_attrs["root"] is True
    assert tool_metric.sum >= 990


def test_no_metrics_when_disabled(
    monkeypatch: pytest.MonkeyPatch, metric_reader: InMemoryMetricReader
) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)

    @mlflow.trace(name="test")
    def test_function() -> str:
        return "result"

    test_function()

    metrics_data = metric_reader.get_metrics_data()

    metric_names = []
    if metrics_data:
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                metric_names.extend(metric.name for metric in scope_metric.metrics)

    assert "mlflow.trace.span.duration" not in metric_names

def test_otel_metrics_import_error_graceful_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import sys
    from mlflow.tracing.processor.otel_metrics_mixin import OtelMetricsMixin

    # 1. Force the OTLP metrics endpoint environment variable to be present
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://localhost:4317")
    
    # 2. Mock out the protocol to trigger the gRPC code path
    monkeypatch.setattr(
        "mlflow.tracing.processor.otel_metrics_mixin._get_otlp_metrics_protocol",
        lambda: "grpc",
    )

    # 3. Simulate a completely missing opentelemetry exporter dependency module
    # We alter sys.modules to raise an ImportError upon accessing it
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.grpc.metric_exporter", None)

    # 4. Instantiate a dummy mixin processor instance to exercise the path
    mixin_instance = OtelMetricsMixin()

    # 5. Clear previous logs captured and fire the critical metrics setup method
    caplog.clear()
    mixin_instance._setup_metrics_if_necessary()

    # 6. Assert that the process didn't crash, returns cleanly, and logged an actionable warning
    assert mixin_instance._duration_histogram is None
    assert any(
        "gRPC OTLP metric exporter is not available" in record.message
        for record in caplog.records
    )
