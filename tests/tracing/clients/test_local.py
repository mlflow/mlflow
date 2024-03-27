from mlflow.entities import SpanStatus, Trace, TraceInfo
from mlflow.tracing.clients import InMemoryTraceClient
from mlflow.tracing.types.constant import TraceStatusCode


def test_log_and_get_trace(monkeypatch, create_trace):
    monkeypatch.setenv("MLFLOW_TRACING_CLIENT_BUFFER_SIZE", "3")

    def _create_trace(request_id: str):
        return Trace(
            trace_info=TraceInfo(
                request_id=request_id,
                experiment_id="test",
                timestamp_ms=0,
                execution_time_ms=1,
                status=SpanStatus(TraceStatusCode.OK),
                request_metadata=[],
                tags=[],
            ),
            trace_data=[],
        )

    client = InMemoryTraceClient.get_instance()
    traces = client.get_traces()
    assert len(traces) == 0

    client.log_trace(create_trace("a"))
    client.log_trace(create_trace("b"))
    client.log_trace(create_trace("c"))

    traces = client.get_traces()
    assert len(traces) == 3
    assert traces[0].trace_info.request_id == "a"

    traces = client.get_traces(1)
    assert len(traces) == 1
    assert traces[0].trace_info.request_id == "c"

    client.log_trace(create_trace("d"))
    traces = client.get_traces()
    assert len(traces) == 3
    assert traces[0].trace_info.request_id == "b"
