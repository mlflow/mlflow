import json
from unittest import mock

import mlflow.tracking.context.default_context
from mlflow.entities.span import LiveSpan
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_TRACKING_USERNAME
from mlflow.tracing.constant import (
    SpanAttributeKey,
    TraceMetadataKey,
)
from mlflow.tracing.processor.mlflow_v3 import MlflowV3SpanProcessor
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import encode_trace_id
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID

from tests.tracing.helper import (
    create_mock_otel_span,
    create_test_trace_info,
    skip_when_testing_trace_sdk,
)


def test_on_start(monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")

    # Root span should create a new trace on start
    trace_id = 12345
    span = create_mock_otel_span(trace_id=trace_id, span_id=1, parent_id=None, start_time=5_000_000)

    processor = MlflowV3SpanProcessor(span_exporter=mock.MagicMock())
    processor.on_start(span)

    # V3 processor uses encoded Otel trace_id as request_id
    request_id = "tr-" + encode_trace_id(trace_id)
    assert len(request_id) == 35  # 3 for "tr-" prefix + 32 for encoded trace_id
    assert span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(request_id)
    assert request_id in InMemoryTraceManager.get_instance()._traces

    # Child span should not create a new trace
    child_span = create_mock_otel_span(
        trace_id=trace_id, span_id=2, parent_id=1, start_time=8_000_000
    )
    processor.on_start(child_span)
    assert child_span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(request_id)


@skip_when_testing_trace_sdk
def test_on_start_during_model_evaluation():
    from mlflow.pyfunc.context import Context, set_prediction_context

    trace_id = 12345
    request_id = "tr-" + encode_trace_id(trace_id)

    # Root span should create a new trace on start
    span = create_mock_otel_span(trace_id=trace_id, span_id=1)
    processor = MlflowV3SpanProcessor(span_exporter=mock.MagicMock())

    with set_prediction_context(Context(request_id=request_id, is_evaluate=True)):
        processor.on_start(span)

    assert span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(request_id)


@skip_when_testing_trace_sdk
def test_on_start_during_run(monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")

    span = create_mock_otel_span(trace_id=12345, span_id=1, parent_id=None, start_time=5_000_000)

    env_experiment_name = "env_experiment_id"
    run_experiment_name = "run_experiment_id"

    mlflow.create_experiment(env_experiment_name)
    run_experiment_id = mlflow.create_experiment(run_experiment_name)

    mlflow.set_experiment(experiment_name=env_experiment_name)
    processor = MlflowV3SpanProcessor(span_exporter=mock.MagicMock())

    with mlflow.start_run(experiment_id=run_experiment_id) as run:
        processor.on_start(span)

        trace_id = "tr-" + encode_trace_id(span.context.trace_id)
        trace = InMemoryTraceManager.get_instance()._traces[trace_id]
        assert trace.info.experiment_id == run_experiment_id
        assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run.info.run_id


def test_on_start_with_experiment_id_override(monkeypatch):
    mlflow.set_experiment(experiment_id=DEFAULT_EXPERIMENT_ID)

    mock_logger = mock.MagicMock()
    monkeypatch.setattr("mlflow.tracing.processor.mlflow_v2._logger", mock_logger)

    processor = MlflowV3SpanProcessor(
        span_exporter=mock.MagicMock(), experiment_id="another_experiment"
    )

    span = create_mock_otel_span(trace_id=123, span_id=1)
    processor.on_start(span)

    trace_id = "tr-" + encode_trace_id(span.context.trace_id)
    trace = InMemoryTraceManager.get_instance()._traces[trace_id]
    assert trace.info.experiment_id == "another_experiment"


def test_on_end():
    trace_info = create_test_trace_info("request_id", 0)
    trace_manager = InMemoryTraceManager.get_instance()
    trace_manager.register_trace("trace_id", trace_info)

    otel_span = create_mock_otel_span(
        name="foo",
        trace_id="trace_id",
        span_id=1,
        parent_id=None,
        start_time=5_000_000,
        end_time=9_000_000,
    )
    span = LiveSpan(otel_span, "request_id")
    span.set_status("OK")
    span.set_inputs({"input1": "very long input" * 100})
    span.set_outputs({"output": "very long output" * 100})

    mock_exporter = mock.MagicMock()
    mock_client = mock.MagicMock()
    mock_client._start_tracked_trace.side_effect = Exception("error")
    processor = MlflowV3SpanProcessor(span_exporter=mock_exporter)

    processor.on_end(otel_span)

    mock_exporter.export.assert_called_once_with((otel_span,))
    # Trace info should be updated according to the span attributes
    assert trace_info.status == TraceStatus.OK
    assert trace_info.execution_time_ms == 4
    assert trace_info.tags == {}

    # Non-root span should not be exported
    mock_exporter.reset_mock()
    child_span = create_mock_otel_span(trace_id="trace_id", span_id=2, parent_id=1)
    processor.on_end(child_span)
    mock_exporter.export.assert_not_called()
