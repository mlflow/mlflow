"""
Tests demonstrating that mlflow.set_session() works universally across all frameworks.

These tests show that session tracking is not Strands-specific but works with
OpenAI, Anthropic, and any framework using MLflow tracing.
"""

import pytest

import mlflow
from mlflow.tracing.constant import TraceMetadataKey


@pytest.mark.skipif(
    not pytest.importorskip("openai", reason="OpenAI not installed"),
    reason="OpenAI not installed",
)
def test_universal_session_tracking_openai(mock_openai):
    mlflow.openai.autolog()
    try:
        # Use mlflow.set_session() with OpenAI
        with mlflow.set_session("openai-session-123"):
            response = mock_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            assert response is not None

        # Verify session ID was captured
        trace_id = mlflow.get_last_active_trace_id()
        assert trace_id is not None

        trace = mlflow.get_trace(trace_id)
        session_metadata = trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)
        assert session_metadata == "openai-session-123", (
            f"Expected 'openai-session-123', got {session_metadata}"
        )
    finally:
        mlflow.openai.autolog(disable=True)


@pytest.mark.skipif(
    not pytest.importorskip("anthropic", reason="Anthropic not installed"),
    reason="Anthropic not installed",
)
def test_universal_session_tracking_anthropic(monkeypatch):
    import anthropic
    from anthropic.types import Message, TextBlock, Usage

    # Mock Anthropic API response
    def mock_create(*args, **kwargs):
        return Message(
            id="msg_123",
            type="message",
            role="assistant",
            content=[TextBlock(type="text", text="Hello! How can I help?")],
            model="claude-3-opus-20240229",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=20),
        )

    monkeypatch.setattr("anthropic.resources.messages.Messages.create", mock_create)

    mlflow.anthropic.autolog()
    try:
        client = anthropic.Anthropic(api_key="test-key")

        # Use mlflow.set_session() with Anthropic
        with mlflow.set_session("claude-session-456"):
            message = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello!"}],
            )

            assert message is not None

        # Verify session ID was captured
        trace_id = mlflow.get_last_active_trace_id()
        assert trace_id is not None

        trace = mlflow.get_trace(trace_id)
        session_metadata = trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)
        assert session_metadata == "claude-session-456", (
            f"Expected 'claude-session-456', got {session_metadata}"
        )
    finally:
        mlflow.anthropic.autolog(disable=True)


def test_universal_session_tracking_custom_code():
    @mlflow.trace
    def custom_agent(user_input: str) -> str:
        """A custom agent function."""
        return f"Response to: {user_input}"

    # Use mlflow.set_session() with custom code
    with mlflow.set_session("custom-session-789"):
        result = custom_agent("What's the weather?")
        assert result == "Response to: What's the weather?"

    # Verify session ID was captured
    trace_id = mlflow.get_last_active_trace_id()
    assert trace_id is not None

    trace = mlflow.get_trace(trace_id)
    session_metadata = trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)
    assert session_metadata == "custom-session-789", (
        f"Expected 'custom-session-789', got {session_metadata}"
    )


def test_universal_session_tracking_mixed_frameworks(mock_openai, monkeypatch):
    # Setup OpenAI
    mlflow.openai.autolog()

    # Setup Anthropic mock
    if pytest.importorskip("anthropic", reason="Anthropic not installed"):
        import anthropic
        from anthropic.types import Message, TextBlock, Usage

        def mock_create(*args, **kwargs):
            return Message(
                id="msg_456",
                type="message",
                role="assistant",
                content=[TextBlock(type="text", text="Claude response")],
                model="claude-3-opus-20240229",
                stop_reason="end_turn",
                usage=Usage(input_tokens=5, output_tokens=10),
            )

        monkeypatch.setattr("anthropic.resources.messages.Messages.create", mock_create)
        mlflow.anthropic.autolog()
        anthropic_client = anthropic.Anthropic(api_key="test-key")

    @mlflow.trace
    def custom_function(text: str) -> str:
        return f"Processed: {text}"

    try:
        # Single session across multiple frameworks
        with mlflow.set_session("multi-framework-session"):
            # Call 1: OpenAI
            mock_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "OpenAI call"}],
            )
            trace1_id = mlflow.get_last_active_trace_id()

            # Call 2: Custom function
            custom_function("Custom call")
            trace2_id = mlflow.get_last_active_trace_id()

            # Call 3: Anthropic (if available)
            if "anthropic_client" in locals():
                anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=100,
                    messages=[{"role": "user", "content": "Claude call"}],
                )
                trace3_id = mlflow.get_last_active_trace_id()

        # Verify all traces have the same session ID
        trace1 = mlflow.get_trace(trace1_id)
        trace2 = mlflow.get_trace(trace2_id)

        assert trace1.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == (
            "multi-framework-session"
        )
        assert trace2.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == (
            "multi-framework-session"
        )

        if "anthropic_client" in locals():
            trace3 = mlflow.get_trace(trace3_id)
            assert trace3.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == (
                "multi-framework-session"
            )

    finally:
        mlflow.openai.autolog(disable=True)
        if pytest.importorskip("anthropic", reason="Anthropic not installed"):
            mlflow.anthropic.autolog(disable=True)


def test_universal_session_auto_generated():
    @mlflow.trace
    def my_function(x: int) -> int:
        return x * 2

    # Use auto-generated session ID
    with mlflow.set_session() as session_id:
        assert session_id is not None
        assert len(session_id) == 32  # UUID hex

        my_function(5)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    captured_session = trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)

    assert captured_session == session_id
    assert len(captured_session) == 32


def test_universal_session_nested_contexts():
    @mlflow.trace
    def outer_function(x: str) -> str:
        return f"Outer: {x}"

    @mlflow.trace
    def inner_function(x: str) -> str:
        return f"Inner: {x}"

    # Outer session
    with mlflow.set_session("outer-session"):
        outer_function("call 1")
        trace1_id = mlflow.get_last_active_trace_id()

        # Inner session (overrides)
        with mlflow.set_session("inner-session"):
            inner_function("call 2")
            trace2_id = mlflow.get_last_active_trace_id()

        # Back to outer session
        outer_function("call 3")
        trace3_id = mlflow.get_last_active_trace_id()

    # Verify session IDs
    trace1 = mlflow.get_trace(trace1_id)
    trace2 = mlflow.get_trace(trace2_id)
    trace3 = mlflow.get_trace(trace3_id)

    assert trace1.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "outer-session"
    assert trace2.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "inner-session"
    assert trace3.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "outer-session"


# Fixtures for mock services


@pytest.fixture(scope="module")
def mock_openai_server():
    """Set up mock OpenAI server."""
    from tests.helper_functions import start_mock_openai_server

    with start_mock_openai_server() as mock_server:
        yield mock_server


@pytest.fixture
def mock_openai(mock_openai_server, monkeypatch):
    """Create OpenAI client pointing to mock server."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE", mock_openai_server)

    import openai

    return openai.OpenAI(api_key="test-key", base_url=mock_openai_server)
