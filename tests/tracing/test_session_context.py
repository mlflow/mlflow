"""
Tests for session context management in MLflow tracing.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import mlflow
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.session_context import (
    _get_session_id_for_trace,
    get_session_id,
    set_session,
    set_session_id,
)


def test_get_session_id_returns_none_when_not_set():
    assert get_session_id() is None


def test_set_and_get_session_id():
    test_id = "test-session-123"
    set_session_id(test_id)
    try:
        assert get_session_id() == test_id
    finally:
        set_session_id(None)


def test_set_session_id_to_none_clears_session():
    set_session_id("test-session")
    set_session_id(None)
    assert get_session_id() is None


def test_internal_get_session_id_for_trace():
    assert _get_session_id_for_trace() is None

    set_session_id("internal-test")
    try:
        assert _get_session_id_for_trace() == "internal-test"
    finally:
        set_session_id(None)


def test_set_session_with_explicit_id():
    with set_session("explicit-session-456") as session_id:
        assert session_id == "explicit-session-456"
        assert get_session_id() == "explicit-session-456"

    # Session ID should be cleared after context
    assert get_session_id() is None


def test_set_session_auto_generates_uuid():
    with set_session() as session_id:
        assert session_id is not None
        # Should be a valid UUID hex string
        assert len(session_id) == 32
        assert get_session_id() == session_id

    assert get_session_id() is None


def test_set_session_auto_generate_false():
    with set_session(auto_generate=False) as session_id:
        assert session_id is None
        assert get_session_id() is None

    assert get_session_id() is None


def test_set_session_nested_contexts():
    with set_session("outer-session"):
        assert get_session_id() == "outer-session"

        with set_session("inner-session") as inner:
            assert get_session_id() == "inner-session"
            assert inner == "inner-session"

        # Should restore outer session
        assert get_session_id() == "outer-session"

    # Should be cleared after all contexts
    assert get_session_id() is None


def test_set_session_restores_on_exception():
    set_session_id("original-session")
    try:
        # Test that session is set during context, then verify it raises
        try:
            with set_session("exception-session"):
                assert get_session_id() == "exception-session"
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Should restore original session after exception
        assert get_session_id() == "original-session"
    finally:
        set_session_id(None)


def test_session_id_is_thread_local():
    results = {}
    barrier = threading.Barrier(3)

    def thread_task(thread_id, session_id):
        set_session_id(session_id)
        barrier.wait()  # Synchronize all threads
        # After sync, read back the session ID
        results[thread_id] = get_session_id()
        set_session_id(None)  # Cleanup

    threads = []
    for i in range(3):
        t = threading.Thread(target=thread_task, args=(i, f"session-thread-{i}"))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Each thread should have its own session ID
    assert results[0] == "session-thread-0"
    assert results[1] == "session-thread-1"
    assert results[2] == "session-thread-2"


def test_set_session_context_manager_thread_local():
    results = {}

    def thread_task(thread_id):
        with set_session(f"ctx-session-{thread_id}") as session:
            results[thread_id] = session

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(thread_task, i) for i in range(5)]
        for f in futures:
            f.result()

    for i in range(5):
        assert results[i] == f"ctx-session-{i}"


def test_trace_captures_session_id_from_context():
    with set_session("trace-session-test") as session_id:

        @mlflow.trace
        def my_function():
            return "result"

        my_function()

        # Get the trace and verify session ID
        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        assert trace is not None
        assert trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == session_id


def test_multiple_traces_share_same_session_id():
    with set_session("multi-trace-session"):

        @mlflow.trace
        def func1():
            return "one"

        @mlflow.trace
        def func2():
            return "two"

        func1()
        trace1_id = mlflow.get_last_active_trace_id()
        func2()
        trace2_id = mlflow.get_last_active_trace_id()

        trace1 = mlflow.get_trace(trace1_id)
        trace2 = mlflow.get_trace(trace2_id)

        assert (
            trace1.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "multi-trace-session"
        )
        assert (
            trace2.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "multi-trace-session"
        )


def test_trace_without_session_has_no_session_metadata():
    @mlflow.trace
    def no_session_func():
        return "no session"

    no_session_func()

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert TraceMetadataKey.TRACE_SESSION not in trace.info.trace_metadata


def test_start_span_captures_session_id():
    with set_session("span-session-test"):
        with mlflow.start_span("test-span") as span:
            span.set_inputs({"key": "value"})

        trace = mlflow.get_trace(span.trace_id)
        assert trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "span-session-test"


def test_set_session_id_function_with_trace():
    set_session_id("function-set-session")
    try:

        @mlflow.trace
        def traced_func():
            return "result"

        traced_func()

        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        assert (
            trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "function-set-session"
        )
    finally:
        set_session_id(None)


def test_session_context_coexists_with_manual_update():
    with set_session("context-session"):

        @mlflow.trace
        def my_func():
            mlflow.update_current_trace(
                metadata={"custom_key": "custom_value"},
            )
            return "result"

        my_func()

        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        # Both should be present
        assert trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "context-session"
        assert trace.info.trace_metadata.get("custom_key") == "custom_value"


def test_manual_session_update_overrides_context():
    with set_session("context-session"):

        @mlflow.trace
        def my_func():
            # Explicitly override the session
            mlflow.update_current_trace(
                metadata={TraceMetadataKey.TRACE_SESSION: "manual-override-session"},
            )
            return "result"

        my_func()

        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        # Manual override should take precedence
        assert (
            trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)
            == "manual-override-session"
        )


@pytest.mark.asyncio
async def test_session_context_with_async_trace():
    with set_session("async-session-test"):

        @mlflow.trace
        async def async_func():
            return "async result"

        await async_func()

        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        assert trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "async-session-test"
