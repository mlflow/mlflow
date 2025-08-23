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


def test_incremental_span_name_deduplication():
    """Test that span names are deduplicated incrementally as spans end."""
    # Reset trace manager to ensure clean state
    InMemoryTraceManager.reset()
    trace_manager = InMemoryTraceManager.get_instance()

    # Create a trace with trace_id=12345
    trace_id = 12345
    request_id = "tr-" + encode_trace_id(trace_id)

    # Create root span and start the trace
    root_span = create_mock_otel_span(
        name="process",
        trace_id=trace_id,
        span_id=1,
        parent_id=None,
        start_time=1_000_000,
        end_time=10_000_000,
    )

    processor = MlflowV3SpanProcessor(span_exporter=mock.MagicMock())

    # Start root span - this creates the trace
    processor.on_start(root_span)

    # Register root span with trace manager
    live_root = LiveSpan(root_span, request_id)
    live_root._request_id = request_id
    trace_manager.register_span(live_root)

    # Create first child span (duplicate "process")
    child1 = create_mock_otel_span(
        name="process",
        trace_id=trace_id,
        span_id=2,
        parent_id=1,
        start_time=2_000_000,
        end_time=3_000_000,
    )
    processor.on_start(child1)
    live_child1 = LiveSpan(child1, request_id)
    live_child1._request_id = request_id
    trace_manager.register_span(live_child1)

    # End first child - should trigger deduplication of the two "process" spans
    processor.on_end(child1)

    with trace_manager.get_trace(request_id) as trace:
        span_names = [span.name for span in trace.span_dict.values()]
        # The two "process" spans should be deduplicated
        assert "process_1" in span_names  # root renamed
        assert "process_2" in span_names  # first child renamed

    # Create and register second child (first "query")
    child2 = create_mock_otel_span(
        name="query",
        trace_id=trace_id,
        span_id=3,
        parent_id=1,
        start_time=4_000_000,
        end_time=5_000_000,
    )
    processor.on_start(child2)
    live_child2 = LiveSpan(child2, request_id)
    live_child2._request_id = request_id
    trace_manager.register_span(live_child2)

    # End second child - no duplicate queries yet
    processor.on_end(child2)

    with trace_manager.get_trace(request_id) as trace:
        span_names = [span.name for span in trace.span_dict.values()]
        assert "query" in span_names  # First query not renamed (no duplicate)

    # Create and register third child (third "process")
    child3 = create_mock_otel_span(
        name="process",
        trace_id=trace_id,
        span_id=4,
        parent_id=1,
        start_time=6_000_000,
        end_time=7_000_000,
    )
    processor.on_start(child3)
    live_child3 = LiveSpan(child3, request_id)
    live_child3._request_id = request_id
    trace_manager.register_span(live_child3)

    # End third child - should update process deduplication
    processor.on_end(child3)

    with trace_manager.get_trace(request_id) as trace:
        span_names = [span.name for span in trace.span_dict.values()]
        # With _original_name tracking, the third process is correctly numbered
        assert "process_1" in span_names  # root still renamed
        assert "process_2" in span_names  # first child still renamed
        assert "process_3" in span_names  # third process correctly renamed

    # Create and register fourth child (second "query")
    child4 = create_mock_otel_span(
        name="query",
        trace_id=trace_id,
        span_id=5,
        parent_id=1,
        start_time=8_000_000,
        end_time=9_000_000,
    )
    processor.on_start(child4)
    live_child4 = LiveSpan(child4, request_id)
    live_child4._request_id = request_id
    trace_manager.register_span(live_child4)

    # End fourth child - should trigger query deduplication
    processor.on_end(child4)

    with trace_manager.get_trace(request_id) as trace:
        span_names = [span.name for span in trace.span_dict.values()]
        # Both queries should now be deduplicated
        assert "query_1" in span_names  # First query renamed
        assert "query_2" in span_names  # Second query renamed

    # Finally end the root span
    processor.on_end(root_span)

    # Final check - with the new _original_name tracking, all spans are correctly deduplicated
    with trace_manager.get_trace(request_id) as trace:
        span_names = sorted([span.name for span in trace.span_dict.values()])
        # All three "process" spans and both "query" spans are correctly numbered
        assert span_names == ["process_1", "process_2", "process_3", "query_1", "query_2"]


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

    # Child spans should be exported
    mock_exporter.reset_mock()
    child_span = create_mock_otel_span(trace_id="trace_id", span_id=2, parent_id=1)
    # Set the REQUEST_ID attribute so the processor can find the trace
    child_span.set_attribute(SpanAttributeKey.REQUEST_ID, json.dumps("request_id"))
    processor.on_end(child_span)
    mock_exporter.export.assert_called_once_with((child_span,))

    # Trace info should be updated according to the span attributes
    manager_trace = trace_manager.pop_trace("trace_id")
    trace_info = manager_trace.trace.info
    assert trace_info.status == TraceStatus.OK
    assert trace_info.execution_time_ms == 4
    assert trace_info.tags == {}
