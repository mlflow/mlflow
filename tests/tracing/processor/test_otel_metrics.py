import os
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


@pytest.mark.flaky(attempts=3, condition=os.name == "nt")
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
    data_points.sort(key=lambda dp: dp.sum)
    llm_metric, chain_metric, tool_metric = data_points

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
    assert tool_metric.sum >= 1000


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
