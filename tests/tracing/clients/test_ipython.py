from unittest.mock import Mock

from mlflow.tracing.clients import get_trace_client


class MockIPython:
    def __init__(self):
        self.execution_count = 0

    def mock_run_cell(self):
        self.execution_count += 1


def test_display_is_not_called_without_ipython(monkeypatch, create_trace):
    # in an IPython environment, the interactive shell will
    # be returned. however, for test purposes, just mock that
    # the value is not None.
    mock_display = Mock()
    monkeypatch.setattr("IPython.display.display", mock_display)

    client = get_trace_client()
    client.log_trace(create_trace("a"))
    assert mock_display.call_count == 0

    monkeypatch.setattr("IPython.get_ipython", lambda: MockIPython())
    client.log_trace(create_trace("b"))
    assert mock_display.call_count == 1


def test_ipython_client_only_logs_once_per_execution(monkeypatch, create_trace):
    mock_ipython = MockIPython()
    monkeypatch.setattr("IPython.get_ipython", lambda: mock_ipython)
    client = get_trace_client()

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
