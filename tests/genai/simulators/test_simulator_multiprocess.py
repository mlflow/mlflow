from __future__ import annotations

import multiprocessing
import uuid
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest

import mlflow
from mlflow.genai.simulators import ConversationSimulator, PredictResult


def _remote_agent_worker(
    correlation_id: str,
    tracking_uri: str,
    experiment_id: str,
    result_queue: multiprocessing.Queue,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_span(name="remote-agent-root", span_type="CHAIN") as span:
        span.set_inputs({"correlation_id": correlation_id, "query": "test"})
        # Embed the correlation_id in the span output so trace-search can find it
        span.set_outputs({"answer": "42", "correlation_id": correlation_id})

    # trace_id is set only after the root span closes
    trace_id = mlflow.get_last_active_trace_id(thread_local=True)
    result_queue.put(trace_id)


@pytest.mark.parametrize(
    "response_format",
    ["predict_result_with_id", "predict_result_with_none", "plain"],
)
def test_invoke_predict_fn_response_formats(response_format, simple_test_case):
    simulator = ConversationSimulator(test_cases=[simple_test_case], max_turns=1)
    external_trace_id = "multiproc-trace-id-abc"

    def _make_response():
        return {
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Result from remote agent"}],
                }
            ]
        }

    if response_format == "predict_result_with_id":

        def predict_fn(input=None, **kwargs):
            return PredictResult(response=_make_response(), trace_id=external_trace_id)

    elif response_format == "predict_result_with_none":

        def predict_fn(input=None, **kwargs):
            return PredictResult(response=_make_response(), trace_id=None)

    else:

        def predict_fn(input=None, **kwargs):
            return _make_response()

    @contextmanager
    def _noop_ctx(**kwargs):
        yield

    with (
        patch("mlflow.tracing.context", side_effect=_noop_ctx),
        patch("mlflow.get_last_active_trace_id", return_value="local-trace-fallback"),
        patch("mlflow.log_expectation"),
    ):
        response, trace_id = simulator._invoke_predict_fn(
            predict_fn=predict_fn,
            input_messages=[{"role": "user", "content": "hello"}],
            trace_session_id="sess-123",
            goal="test goal",
            persona="tester",
            simulation_guidelines=None,
            context={},
            expectations=None,
            turn=0,
        )

    assert response is not None

    if response_format == "predict_result_with_id":
        assert trace_id == external_trace_id
    else:
        assert trace_id == "local-trace-fallback"


def test_multiprocess_remote_agent_trace_retrieval(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"multiprocess-test-{uuid.uuid4().hex[:8]}"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    # Correlation ID that ties the predict_fn call to the remote trace.
    correlation_id = uuid.uuid4().hex

    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_remote_agent_worker,
        args=(correlation_id, tracking_uri, experiment_id, result_queue),
    )
    proc.start()
    proc.join(timeout=30)
    assert proc.exitcode == 0, "Remote agent process failed"

    remote_trace_id: str = result_queue.get_nowait()
    assert remote_trace_id is not None

    # Verify the trace was actually written by the child process.
    mlflow.flush_trace_async_logging()
    from mlflow.tracking.client import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    traces = client.search_traces(
        locations=[experiment_id],
        filter_string=f"trace.text LIKE '%{correlation_id}%'",
    )
    assert len(traces) == 1
    assert traces[0].info.trace_id == remote_trace_id

    # predict_fn uses PredictResult to hand back the externally-obtained trace ID.
    def predict_fn(input=None, **kwargs):
        response = {"output": [{"role": "assistant", "content": "42"}]}
        return PredictResult(response=response, trace_id=remote_trace_id)

    fetched_trace_ids = []

    with (
        patch("mlflow.genai.simulators.simulator.invoke_model_without_tracing") as mock_invoke,
        patch("mlflow.tracing.client.TracingClient") as mock_tc_cls,
        patch("mlflow.flush_trace_async_logging"),
    ):
        mock_invoke.side_effect = [
            "Test message",
            '{"rationale": "Goal achieved!", "result": "yes"}',
        ]
        mock_trace = Mock()
        mock_trace.info.trace_metadata = {}
        mock_trace.info.assessments = []

        def _get_trace(tid):
            fetched_trace_ids.append(tid)
            return mock_trace

        mock_tc_cls.return_value = Mock(get_trace=_get_trace)

        simulator = ConversationSimulator(
            test_cases=[{"goal": "Test remote agent"}],
            max_turns=1,
        )
        with mlflow.start_run():
            all_traces = simulator.simulate(predict_fn)

    assert len(all_traces) == 1
    assert len(all_traces[0]) == 1
    # TracingClient.get_trace must have been called with the remote trace ID.
    assert fetched_trace_ids == [remote_trace_id]
