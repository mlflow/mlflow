from mlflow.tracing.clients import InMemoryTraceClient
from mlflow.tracing.types.model import Status, StatusCode, Trace, TraceInfo


def test_log_and_get_trace(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACING_CLIENT_BUFFER_SIZE", "3")

    def _create_trace(trace_id: str):
        return Trace(
            trace_info=TraceInfo(
                trace_id=trace_id,
                experiment_id="test",
                start_time=0,
                end_time=1,
                status=Status(StatusCode.OK),
                attributes={},
                tags={},
            ),
            trace_data=[],
        )

    client = InMemoryTraceClient.get_instance()
    traces = client.get_traces()
    assert len(traces) == 0

    client.log_trace(_create_trace("a"))
    client.log_trace(_create_trace("b"))
    client.log_trace(_create_trace("c"))

    traces = client.get_traces()
    assert len(traces) == 3
    assert traces[0].trace_info.trace_id == "a"

    traces = client.get_traces(1)
    assert len(traces) == 1
    assert traces[0].trace_info.trace_id == "c"

    client.log_trace(_create_trace("d"))
    traces = client.get_traces()
    assert len(traces) == 3
    assert traces[0].trace_info.trace_id == "b"
