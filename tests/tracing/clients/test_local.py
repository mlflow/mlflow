from mlflow.tracing.clients import InMemoryTraceClient

from tests.tracing.helper import create_trace


def test_log_and_get_trace(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACING_CLIENT_BUFFER_SIZE", "3")

    client = InMemoryTraceClient.get_instance()
    traces = client.get_traces()
    assert len(traces) == 0

    client.log_trace(create_trace("a"))
    client.log_trace(create_trace("b"))
    client.log_trace(create_trace("c"))

    traces = client.get_traces()
    assert len(traces) == 3
    assert traces[0].info.request_id == "a"

    traces = client.get_traces(1)
    assert len(traces) == 1
    assert traces[0].info.request_id == "c"

    client.log_trace(create_trace("d"))
    traces = client.get_traces()
    assert len(traces) == 3
    assert traces[0].info.request_id == "b"
