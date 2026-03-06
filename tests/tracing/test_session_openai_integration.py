"""
Tests for OpenAI autologging integration with session context.

These tests verify that when autologging is enabled for OpenAI,
traces automatically capture the session ID from the active context.
"""

import pytest

import mlflow
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.session_context import set_session


@pytest.fixture
def mock_openai(monkeypatch):
    """Set up mock OpenAI server using MLflow's test infrastructure."""
    from tests.helper_functions import start_mock_openai_server

    with start_mock_openai_server() as mock_server:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_BASE", mock_server)

        import openai

        client = openai.OpenAI(api_key="test-key", base_url=mock_server)
        yield client


@pytest.mark.skipif(
    not pytest.importorskip("openai", reason="OpenAI not installed"),
    reason="OpenAI not installed",
)
def test_openai_autolog_captures_session_id(mock_openai):
    mlflow.openai.autolog()
    try:
        with set_session("openai-test-session") as session_id:
            response = mock_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            # Verify response was received
            assert response is not None

            # Get the trace and verify session ID
            trace_id = mlflow.get_last_active_trace_id()
            assert trace_id is not None, "No trace was created by autologging"

            trace = mlflow.get_trace(trace_id)
            assert trace is not None, "Could not retrieve trace"

            session_metadata = trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)
            assert session_metadata == session_id, (
                f"Expected session ID {session_id}, got {session_metadata}"
            )
    finally:
        mlflow.openai.autolog(disable=True)


@pytest.mark.skipif(
    not pytest.importorskip("openai", reason="OpenAI not installed"),
    reason="OpenAI not installed",
)
def test_openai_multiple_calls_same_session(mock_openai):
    mlflow.openai.autolog()
    try:
        with set_session("multi-call-session"):
            # First call
            mock_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "First message"}],
            )
            trace1_id = mlflow.get_last_active_trace_id()

            # Second call
            mock_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Second message"}],
            )
            trace2_id = mlflow.get_last_active_trace_id()

            trace1 = mlflow.get_trace(trace1_id)
            trace2 = mlflow.get_trace(trace2_id)

            # Both traces should have the same session ID
            assert (
                trace1.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)
                == "multi-call-session"
            )
            assert (
                trace2.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)
                == "multi-call-session"
            )
    finally:
        mlflow.openai.autolog(disable=True)


@pytest.mark.skipif(
    not pytest.importorskip("openai", reason="OpenAI not installed"),
    reason="OpenAI not installed",
)
def test_openai_autolog_without_session(mock_openai):
    mlflow.openai.autolog()
    try:
        # No session set
        response = mock_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "No session message"}],
        )

        assert response is not None

        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        assert trace is not None

        # Should not have session metadata
        assert TraceMetadataKey.TRACE_SESSION not in trace.info.trace_metadata
    finally:
        mlflow.openai.autolog(disable=True)
