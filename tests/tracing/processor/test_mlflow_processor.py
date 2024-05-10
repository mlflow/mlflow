import json
import time
from unittest import mock

import pytest

import mlflow.tracking.context.default_context
from mlflow.entities.span import LiveSpan
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_TRACKING_USERNAME
from mlflow.pyfunc.context import Context, set_prediction_context
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey, TraceTagKey
from mlflow.tracing.processor.mlflow import MlflowSpanProcessor
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import encode_trace_id
from mlflow.utils.os import is_windows

from tests.tracing.helper import create_mock_otel_span, create_test_trace_info

_TRACE_ID = 12345
_REQUEST_ID = f"tr-{_TRACE_ID}"


def test_on_start(clear_singleton, monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")

    # Root span should create a new trace on start
    span = create_mock_otel_span(
        trace_id=_TRACE_ID, span_id=1, parent_id=None, start_time=5_000_000
    )
    trace_info = create_test_trace_info(_REQUEST_ID, 0)

    mock_client = mock.MagicMock()
    mock_client._start_tracked_trace.return_value = trace_info
    processor = MlflowSpanProcessor(span_exporter=mock.MagicMock(), client=mock_client)

    processor.on_start(span)

    mock_client._start_tracked_trace.assert_called_once_with(
        experiment_id="0",
        timestamp_ms=5,
        request_metadata={},
        tags={"mlflow.user": "bob", "mlflow.source.name": "test", "mlflow.source.type": "LOCAL"},
    )
    assert span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(_REQUEST_ID)
    assert _REQUEST_ID in InMemoryTraceManager.get_instance()._traces

    # Child span should not create a new trace
    child_span = create_mock_otel_span(
        trace_id=_TRACE_ID, span_id=2, parent_id=1, start_time=8_000_000
    )
    mock_client._start_tracked_trace.reset_mock()
    processor.on_start(child_span)

    mock_client._start_tracked_trace.assert_not_called()
    assert child_span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(_REQUEST_ID)


@pytest.mark.skipif(is_windows(), reason="Timestamp is not precise enough on Windows")
def test_on_start_adjust_span_timestamp_to_exclude_backend_latency(clear_singleton, monkeypatch):
    monkeypatch.setenv("MLFLOW_TESTING", "false")
    trace_info = create_test_trace_info(_REQUEST_ID, 0)
    mock_client = mock.MagicMock()

    def _mock_start_tracked_trace():
        time.sleep(0.5)  # Simulate backend latency
        return trace_info

    mock_client._start_tracked_trace.side_effect = _mock_start_tracked_trace
    processor = MlflowSpanProcessor(span_exporter=mock.MagicMock(), client=mock_client)

    original_start_time = time.time_ns()
    span = create_mock_otel_span(trace_id=_TRACE_ID, span_id=1, start_time=original_start_time)

    # make sure _start_tracked_trace is invoked
    assert processor._trace_manager.get_request_id_from_trace_id(span.context.trace_id) is None
    processor.on_start(span)

    assert span.start_time > original_start_time
    # The span timestamp should not include the backend latency (0.5 second)
    assert time.time_ns() - span.start_time < 100_000_000  # 0.1 second


def test_on_start_with_experiment_id(clear_singleton, monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")

    experiment_id = "test_experiment_id"
    span = create_mock_otel_span(
        trace_id=_TRACE_ID, span_id=1, parent_id=None, start_time=5_000_000
    )
    span.set_attribute(SpanAttributeKey.EXPERIMENT_ID, json.dumps(experiment_id))
    trace_info = create_test_trace_info(_REQUEST_ID, experiment_id=experiment_id)

    mock_client = mock.MagicMock()
    mock_client._start_tracked_trace.return_value = trace_info
    processor = MlflowSpanProcessor(span_exporter=mock.MagicMock(), client=mock_client)

    processor.on_start(span)

    mock_client._start_tracked_trace.assert_called_once_with(
        experiment_id=experiment_id,
        timestamp_ms=5,
        request_metadata={},
        tags={"mlflow.user": "bob", "mlflow.source.name": "test", "mlflow.source.type": "LOCAL"},
    )
    assert span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(_REQUEST_ID)
    assert _REQUEST_ID in InMemoryTraceManager.get_instance()._traces


def test_on_start_fallback_to_client_side_request_id(clear_singleton, monkeypatch):
    monkeypatch.setenv("MLFLOW_TESTING", "false")
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")
    span = create_mock_otel_span(
        trace_id=_TRACE_ID, span_id=1, parent_id=None, start_time=5_000_000
    )

    mock_client = mock.MagicMock()
    mock_client._start_tracked_trace.side_effect = Exception("error")
    processor = MlflowSpanProcessor(span_exporter=mock.MagicMock(), client=mock_client)

    processor.on_start(span)

    mock_client._start_tracked_trace.assert_called_once_with(
        experiment_id="0",
        timestamp_ms=5,
        request_metadata={},
        tags={"mlflow.user": "bob", "mlflow.source.name": "test", "mlflow.source.type": "LOCAL"},
    )
    # When the backend returns an error, the request_id is generated at client side from trace_id
    expected_request_id = encode_trace_id(_TRACE_ID)
    assert span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(expected_request_id)
    with InMemoryTraceManager.get_instance().get_trace(expected_request_id) as trace:
        assert trace.info.experiment_id == "0"
        assert trace.info.timestamp_ms == 5
        assert trace.info.execution_time_ms is None
        assert trace.info.status == TraceStatus.IN_PROGRESS


def test_on_start_during_model_evaluation(clear_singleton):
    # Root span should create a new trace on start
    span = create_mock_otel_span(trace_id=_TRACE_ID, span_id=1)
    mock_client = mock.MagicMock()
    mock_client._start_tracked_trace.return_value = create_test_trace_info(_REQUEST_ID, 0)
    processor = MlflowSpanProcessor(span_exporter=mock.MagicMock(), client=mock_client)

    with set_prediction_context(Context(request_id=_REQUEST_ID, is_evaluate=True)):
        processor.on_start(span)

    mock_client._start_tracked_trace.assert_called_once()
    assert span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(_REQUEST_ID)


def test_on_start_during_run(clear_singleton, monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")

    span = create_mock_otel_span(
        trace_id=_TRACE_ID, span_id=1, parent_id=None, start_time=5_000_000
    )

    env_experiment_name = "env_experiment_id"
    run_experiment_name = "run_experiment_id"

    mlflow.create_experiment(env_experiment_name)
    run_experiment_id = mlflow.create_experiment(run_experiment_name)

    mlflow.set_experiment(experiment_name=env_experiment_name)
    trace_info = create_test_trace_info(_REQUEST_ID)
    mock_client = mock.MagicMock()
    mock_client._start_tracked_trace.return_value = trace_info
    processor = MlflowSpanProcessor(span_exporter=mock.MagicMock(), client=mock_client)

    with mlflow.start_run(experiment_id=run_experiment_id) as run:
        processor.on_start(span)
        expected_run_id = run.info.run_id

    mock_client._start_tracked_trace.assert_called_once_with(
        # expect experiment id to be from the run, not from the environment
        experiment_id=run_experiment_id,
        timestamp_ms=5,
        # expect run id to be set
        request_metadata={"mlflow.sourceRun": expected_run_id},
        tags=mock.ANY,
    )


def test_on_end(clear_singleton):
    trace_info = create_test_trace_info(_REQUEST_ID, 0)
    trace_manager = InMemoryTraceManager.get_instance()
    trace_manager.register_trace(_TRACE_ID, trace_info)

    otel_span = create_mock_otel_span(
        name="foo",
        trace_id=_TRACE_ID,
        span_id=1,
        parent_id=None,
        start_time=5_000_000,
        end_time=9_000_000,
    )
    span = LiveSpan(otel_span, request_id=_REQUEST_ID)
    span.set_status("OK")
    span.set_inputs({"input1": "very long input" * 100})
    span.set_outputs({"output": "very long output" * 100})

    mock_exporter = mock.MagicMock()
    mock_client = mock.MagicMock()
    mock_client._start_tracked_trace.side_effect = Exception("error")
    processor = MlflowSpanProcessor(span_exporter=mock_exporter, client=mock_client)

    processor.on_end(otel_span)

    mock_exporter.export.assert_called_once_with((otel_span,))
    # Trace info should be updated according to the span attributes
    assert trace_info.status == TraceStatus.OK
    assert trace_info.execution_time_ms == 4
    trace_input = trace_info.request_metadata.get(TraceMetadataKey.INPUTS)
    assert len(trace_input) == 250
    assert trace_input.startswith('{"input1": "very long input')
    trace_output = trace_info.request_metadata.get(TraceMetadataKey.OUTPUTS)
    assert len(trace_output) == 250
    assert trace_output.startswith('{"output": "very long output')
    assert trace_info.tags == {TraceTagKey.TRACE_NAME: "foo"}

    # Non-root span should not be exported
    mock_exporter.reset_mock()
    child_span = create_mock_otel_span(trace_id=_TRACE_ID, span_id=2, parent_id=1)
    processor.on_end(child_span)
    mock_exporter.export.assert_not_called()
