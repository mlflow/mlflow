import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.trace_manager import _Trace
from mlflow.tracing.utils.cache import TTLCacheWithLogging


def _mock_span(span_id, parent_id=None):
    span = mock.Mock()
    span.span_id = span_id
    span.parent_id = parent_id
    return span


def test_expire_traces(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_BUFFER_TTL_SECONDS", "1")
    cache = TTLCacheWithLogging(maxsize=1, ttl=1)

    span_1_1 = _mock_span("span_1")
    span_1_2 = _mock_span("span_2", parent_id="span_1")
    cache["tr_1"] = _Trace(None, span_dict={"span_1": span_1_1, "span_2": span_1_2})

    time.sleep(1)

    cache.get("tr-2")  # Accessing any item in the cache should trigger expiration
    time.sleep(1)  # Wait for the expiration to complete

    assert "tr_1" not in cache
    span_1_1.end.assert_called_once()
    span_1_1.set_status.assert_called_once_with(SpanStatusCode.ERROR)
    span_1_1.add_event.assert_called_once()
    event = span_1_1.add_event.call_args[0][0]
    assert isinstance(event, SpanEvent)
    assert event.name == "exception"
    assert event.attributes["exception.message"].startswith("This trace is automatically halted")

    # Non-root span should not be touched
    span_1_2.assert_not_called()


def test_expire_traces_no_expired_trace(monkeypatch):
    cache = TTLCacheWithLogging(maxsize=1, ttl=3600)

    span = _mock_span("span_1")
    cache["tr_1"] = _Trace(None, span_dict={"span": span})

    cache.get("tr-2")
    assert "tr_1" in cache
    span.end.assert_not_called()


def test_expire_traces_empty_cache(monkeypatch):
    cache = TTLCacheWithLogging(maxsize=1, ttl=3600)
    cache.expire()
    assert len(cache) == 0


def test_expire_traces_handle_ttl_update(monkeypatch):
    cache = TTLCacheWithLogging(maxsize=1, ttl=3600)

    # Update TTL after cache creation
    monkeypatch.setenv("MLFLOW_TRACE_BUFFER_TTL_SECONDS", "1")

    span = _mock_span("span_1")
    cache["tr_1"] = _Trace(None, span_dict={"span": span})

    time.sleep(1)
    cache.get("tr-2")
    time.sleep(1)

    assert "tr_1" not in cache
    span.end.assert_called_once()


def test_expire_traces_thread_safe(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_BUFFER_TTL_SECONDS", "1")
    cache = TTLCacheWithLogging(maxsize=1, ttl=1)

    span = _mock_span("span_1")
    span.end.side_effect = lambda: time.sleep(1)  # Simulate slow span export
    cache["tr_1"] = _Trace(None, span_dict={"span": span})

    time.sleep(1)

    def access_cache():
        cache.get("tr-1")

    with ThreadPoolExecutor(max_workers=10) as executor:
        for _ in range(10):
            executor.submit(access_cache)

    time.sleep(3)

    assert "tr_1" not in cache
    # Span should only be expired once
    span.end.assert_called_once()


def test_expire_traces_with_blocking(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_BUFFER_TTL_SECONDS", "1")
    cache = TTLCacheWithLogging(maxsize=1, ttl=1)

    span = _mock_span("span_1")
    span.end.side_effect = lambda: time.sleep(1)  # Simulate slow span export
    cache["tr_1"] = _Trace(None, span_dict={"span": span})

    time.sleep(1)

    cache.expire(block=True)
    assert "tr_1" not in cache
    span.end.assert_called_once()
