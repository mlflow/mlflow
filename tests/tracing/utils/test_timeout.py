import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest

import mlflow
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.export.inference_table import _TRACE_BUFFER, pop_trace
from mlflow.tracing.trace_manager import _Trace
from mlflow.tracing.utils.timeout import MlflowTraceTimeoutCache

from tests.tracing.helper import get_traces, skip_when_testing_trace_sdk


def _mock_span(span_id, parent_id=None):
    span = mock.Mock()
    span.span_id = span_id
    span.parent_id = parent_id
    return span


@pytest.fixture
def cache():
    timeout_cache = MlflowTraceTimeoutCache(timeout=1, maxsize=10)
    yield timeout_cache
    timeout_cache.clear()


def test_expire_traces(cache):
    span_1_1 = _mock_span("span_1")
    span_1_2 = _mock_span("span_2", parent_id="span_1")
    cache["tr_1"] = _Trace(None, span_dict={"span_1": span_1_1, "span_2": span_1_2})
    for _ in range(5):
        if "tr_1" not in cache:
            break
        time.sleep(1)
    else:
        pytest.fail("Trace should be expired within 5 seconds")

    span_1_1.end.assert_called_once()
    span_1_1.set_status.assert_called_once_with(SpanStatusCode.ERROR)
    span_1_1.add_event.assert_called_once()
    event = span_1_1.add_event.call_args[0][0]
    assert isinstance(event, SpanEvent)
    assert event.name == "exception"
    assert event.attributes["exception.message"].startswith("Trace tr_1 is timed out")

    # Non-root span should not be touched
    span_1_2.assert_not_called()


class _SlowModel:
    @mlflow.trace
    def predict(self, x):
        for _ in range(x):
            self.slow_function()
        return

    @mlflow.trace
    def slow_function(self):
        time.sleep(1)


def test_trace_halted_after_timeout(monkeypatch):
    # When MLFLOW_TRACE_TIMEOUT_SECONDS is set, MLflow should halt the trace after
    # the timeout and log it to the backend with an error status
    monkeypatch.setenv("MLFLOW_TRACE_TIMEOUT_SECONDS", "3")

    _SlowModel().predict(5)  # takes 5 seconds

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.execution_time_ms >= 2900  # Some margin for windows
    assert trace.info.status == SpanStatusCode.ERROR
    assert len(trace.data.spans) >= 3

    root_span = trace.data.spans[0]
    assert root_span.name == "predict"
    assert root_span.status.status_code == SpanStatusCode.ERROR
    assert root_span.events[0].name == "exception"
    assert (
        root_span.events[0]
        .attributes["exception.message"]
        .startswith(f"Trace {trace.info.request_id} is timed out")
    )

    first_span = trace.data.spans[1]
    assert first_span.name == "slow_function_1"
    assert first_span.status.status_code == SpanStatusCode.OK

    # The rest of the spans should not be logged to the backend.
    in_progress_traces = mlflow.search_traces(
        filter_string="status = 'IN_PROGRESS'",
        return_type="list",
    )
    assert len(in_progress_traces) == 0


@skip_when_testing_trace_sdk
def test_trace_halted_after_timeout_in_model_serving(
    monkeypatch, mock_databricks_serving_with_tracing_env
):
    from mlflow.pyfunc.context import Context, set_prediction_context

    monkeypatch.setenv("MLFLOW_TRACE_TIMEOUT_SECONDS", "3")

    # Simulate model serving env where multiple requests are processed concurrently
    def _run_single(request_id, seconds):
        with set_prediction_context(Context(request_id=request_id)):
            _SlowModel().predict(seconds)

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(_run_single, ["request-id-1", "request-id-2", "request-id-3"], [5, 6, 1])

    # All traces should be logged
    assert len(_TRACE_BUFFER) == 3

    # Long operation should be halted
    assert pop_trace(request_id="request-id-1")["info"]["state"] == SpanStatusCode.ERROR
    assert pop_trace(request_id="request-id-2")["info"]["state"] == SpanStatusCode.ERROR

    # Short operation should complete successfully
    assert pop_trace(request_id="request-id-3")["info"]["state"] == SpanStatusCode.OK


def test_handle_timeout_update(monkeypatch):
    # Create a first trace. At this moment, there is no timeout set
    _SlowModel().predict(3)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == SpanStatusCode.OK

    # Update timeout env var after cache creation
    monkeypatch.setenv("MLFLOW_TRACE_TIMEOUT_SECONDS", "1")

    # Create a second trace. This should use the new timeout
    _SlowModel().predict(3)

    traces = get_traces()
    assert len(traces) == 2
    assert traces[0].info.status == SpanStatusCode.ERROR

    # Update timeout to a larger value. Trace should complete successfully
    monkeypatch.setenv("MLFLOW_TRACE_TIMEOUT_SECONDS", "100")
    _SlowModel().predict(3)

    traces = get_traces()
    assert len(traces) == 3
    assert traces[0].info.status == SpanStatusCode.OK
