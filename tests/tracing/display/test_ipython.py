import json
from collections import defaultdict
from unittest import mock
from unittest.mock import Mock

import pytest

import mlflow
from mlflow.tracing.display import IPythonTraceDisplayHandler, get_display_handler

from tests.tracing.helper import create_trace


class MockEventRegistry:
    def __init__(self):
        self.events = defaultdict(list)

    def register(self, event, callback):
        self.events[event].append(callback)

    def trigger(self, event):
        for callback in self.events[event]:
            callback(None)


class MockIPython:
    def __init__(self):
        self.events = MockEventRegistry()

    def mock_run_cell(self):
        self.events.trigger("post_run_cell")


@pytest.fixture
def _in_databricks(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.x")


in_databricks = pytest.mark.usefixtures(_in_databricks.__name__)


@in_databricks
def test_display_is_not_called_without_ipython(monkeypatch):
    # in an IPython environment, the interactive shell will
    # be returned. however, for test purposes, just mock that
    # the value is not None.
    mock_display = Mock()
    monkeypatch.setattr("IPython.display.display", mock_display)
    handler = get_display_handler()

    handler.display_traces([create_trace("a")])
    assert mock_display.call_count == 0

    mock_ipython = MockIPython()
    monkeypatch.setattr("IPython.get_ipython", lambda: mock_ipython)

    # reset the singleton so the handler
    # can register the post-display hook
    IPythonTraceDisplayHandler._instance = None
    handler = get_display_handler()
    handler.display_traces([create_trace("b")])

    # simulate cell execution
    mock_ipython.mock_run_cell()

    assert mock_display.call_count == 1


@in_databricks
def test_ipython_client_clears_display_after_execution(monkeypatch):
    mock_ipython = MockIPython()
    monkeypatch.setattr("IPython.get_ipython", lambda: mock_ipython)
    handler = get_display_handler()

    mock_display_handle = Mock()
    mock_display = Mock(return_value=mock_display_handle)
    monkeypatch.setattr("IPython.display.display", mock_display)
    handler.display_traces([create_trace("a")])
    handler.display_traces([create_trace("b")])
    handler.display_traces([create_trace("c")])

    mock_ipython.mock_run_cell()
    # despite many calls to `display_traces`,
    # there should only be one call to `display`
    assert mock_display.call_count == 1

    mock_ipython.mock_run_cell()
    # expect that display is not called,
    # since no traces should be present
    assert mock_display.call_count == 1


@in_databricks
def test_display_is_called_in_correct_functions(monkeypatch):
    mock_ipython = MockIPython()
    monkeypatch.setattr("IPython.get_ipython", lambda: mock_ipython)
    mock_display_handle = Mock()
    mock_display = Mock(return_value=mock_display_handle)
    monkeypatch.setattr("IPython.display.display", mock_display)

    @mlflow.trace
    def foo():
        return 3

    # display should be called after trace creation
    foo()
    mock_ipython.mock_run_cell()
    assert mock_display.call_count == 1

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return [create_trace("a"), create_trace("b"), create_trace("c")]

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)
    mlflow.search_traces(["123"])
    mock_ipython.mock_run_cell()

    assert mock_display.call_count == 2


@in_databricks
def test_display_deduplicates_traces(monkeypatch):
    mock_ipython = MockIPython()
    monkeypatch.setattr("IPython.get_ipython", lambda: mock_ipython)
    handler = get_display_handler()

    mock_display = Mock()
    monkeypatch.setattr("IPython.display.display", mock_display)

    trace_a = create_trace("a")
    trace_b = create_trace("b")
    trace_c = create_trace("c")

    # The display client should dedupe traces to display and only display 3 (not 6).
    handler.display_traces([trace_a])
    handler.display_traces([trace_b])
    handler.display_traces([trace_c])
    handler.display_traces([trace_a, trace_b, trace_c])
    mock_ipython.mock_run_cell()

    expected = [trace_a, trace_b, trace_c]

    assert mock_display.call_count == 1
    assert mock_display.call_args[0][0] == {
        "application/databricks.mlflow.trace": json.dumps(
            [json.loads(t._serialize_for_mimebundle()) for t in expected]
        ),
        "text/plain": repr(expected),
    }


@in_databricks
def test_display_respects_max_limit(monkeypatch):
    mock_ipython = MockIPython()
    monkeypatch.setattr("IPython.get_ipython", lambda: mock_ipython)
    handler = get_display_handler()

    mock_display = Mock()
    monkeypatch.setattr("IPython.display.display", mock_display)

    monkeypatch.setenv("MLFLOW_MAX_TRACES_TO_DISPLAY_IN_NOTEBOOK", 1)

    trace_a = create_trace("a")
    trace_b = create_trace("b")
    trace_c = create_trace("c")
    handler.display_traces([trace_a, trace_b, trace_c])
    mock_ipython.mock_run_cell()

    assert mock_display.call_count == 1
    assert mock_display.call_args[0][0] == {
        "application/databricks.mlflow.trace": trace_a._serialize_for_mimebundle(),
        "text/plain": repr(trace_a),
    }


def test_display_traces_if_not_in_databricks():
    with mock.patch("IPython.get_ipython") as mock_get_ipython:
        handler = get_display_handler()
        handler.display_traces([create_trace("a")])
        mock_get_ipython.assert_not_called()
