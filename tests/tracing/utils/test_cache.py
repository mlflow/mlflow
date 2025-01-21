import time
from unittest import mock

import pytest

from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.trace_manager import _Trace
from mlflow.tracing.utils.cache import MLflowTraceTimeoutCache


def _mock_span(span_id, parent_id=None):
    span = mock.Mock()
    span.span_id = span_id
    span.parent_id = parent_id
    return span


@pytest.fixture
def set_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_TIMEOUT_CHECK_INTERVAL_SECONDS", "1")


@pytest.fixture
def cache():
    timeout_cache = MLflowTraceTimeoutCache(timeout=1, maxsize=10)
    yield timeout_cache
    timeout_cache.clear()


def test_expire_traces(cache):
    span_1_1 = _mock_span("span_1")
    span_1_2 = _mock_span("span_2", parent_id="span_1")
    cache["tr_1"] = _Trace(None, span_dict={"span_1": span_1_1, "span_2": span_1_2})
    time.sleep(2)

    assert "tr_1" not in cache
    span_1_1.end.assert_called_once()
    span_1_1.set_status.assert_called_once_with(SpanStatusCode.ERROR)
    span_1_1.add_event.assert_called_once()
    event = span_1_1.add_event.call_args[0][0]
    assert isinstance(event, SpanEvent)
    assert event.name == "exception"
    assert event.attributes["exception.message"].startswith("Trace tr_1 is automatically halted")

    # Non-root span should not be touched
    span_1_2.assert_not_called()


def test_expire_traces_timeout_update(monkeypatch, cache):
    cache._timeout = 3600

    # Update timeout env var after cache creation
    monkeypatch.setenv("MLFLOW_TRACE_TIMEOUT_SECONDS", "1")
    time.sleep(2)

    span = _mock_span("span_1")
    cache["tr_1"] = _Trace(None, span_dict={"span": span})

    time.sleep(2)

    assert "tr_1" not in cache
    span.end.assert_called_once()


@mock.patch("mlflow.tracing.utils.cache._logger")
def test_expire_traces_timeout_update_warn_when_traces_exist(mock_logger, monkeypatch, cache):
    cache._timeout = 3600

    span = _mock_span("span_1")
    cache["tr_1"] = _Trace(None, span_dict={"span": span})

    # Update timeout env var while there are non-expired traces in the cache
    monkeypatch.setenv("MLFLOW_TRACE_TIMEOUT_SECONDS", "1")

    time.sleep(2)

    mock_logger.warning.assert_called_once()
    warns = mock_logger.warning.call_args_list[0][0]
    assert warns[0].startswith("The timeout of the trace buffer has been updated")
