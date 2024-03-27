from unittest.mock import Mock

from mlflow.tracing.clients import InMemoryTraceClient, IPythonTraceClient, get_trace_client


class MockIPython:
    def __init__(self):
        self.execution_count = 0

    def mock_run_cell(self):
        self.execution_count += 1


def test_get_trace_client_returns_correct_client(monkeypatch):
    assert isinstance(get_trace_client(), InMemoryTraceClient)

    # in an IPython environment, the interactive shell will
    # be returned. however, for test purposes, just mock that
    # the value is not None.
    monkeypatch.setattr("IPython.get_ipython", lambda: True)
    assert isinstance(get_trace_client(), IPythonTraceClient)


def test_ipython_client_only_logs_once_per_execution(monkeypatch, create_trace):
    mock_ipython = MockIPython()
    monkeypatch.setattr("IPython.get_ipython", lambda: mock_ipython)
    client = get_trace_client()
    assert isinstance(client, IPythonTraceClient)

    mock_display_handle = Mock()
    mock_display = Mock(return_value=mock_display_handle)
    monkeypatch.setattr("IPython.display.display", mock_display)
    client.log_trace(create_trace("a"))
    client.log_trace(create_trace("b"))
    client.log_trace(create_trace("c"))

    # there should be one display and two updates
    assert mock_display.call_count == 1
    assert mock_display_handle.update.call_count == 2

    # after incrementing the execution count,
    # the next log should call display again
    mock_ipython.mock_run_cell()
    client.log_trace(create_trace("a"))
    assert mock_display.call_count == 2
