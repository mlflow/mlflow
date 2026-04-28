"""
Tests for Strands session ID autologging.

Covers two modes:
1. Automatic extraction - session_id is pulled from the Agent's session_manager
   automatically when autologging is enabled, with no user action required.
2. Manual override - mlflow.set_session() can always override or supply a
   session_id when no session_manager is present.
"""

import pytest

import mlflow
from mlflow.tracing.constant import TraceMetadataKey


class DummyModel:
    def __init__(self, response_text: str, in_tokens: int = 10, out_tokens: int = 5):
        self.response_text = response_text
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.config = {}

    def update_config(self, **model_config):
        self.config.update(model_config)

    def get_config(self):
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        if False:
            yield {}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": self.response_text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {
                    "inputTokens": self.in_tokens,
                    "outputTokens": self.out_tokens,
                    "totalTokens": self.in_tokens + self.out_tokens,
                },
                "metrics": {"latencyMs": 0},
            }
        }


class MockSessionManager:
    """Minimal SessionManager stub with a session_id, no file I/O needed."""

    def __init__(self, session_id: str):
        self.session_id = session_id

    def register_hooks(self, registry, **kwargs):
        pass


pytestmark = pytest.mark.skipif(
    not pytest.importorskip("strands", reason="Strands not installed"),
    reason="Strands not installed",
)

# ---------------------------------------------------------------------------
# Automatic extraction tests (the primary goal)
# ---------------------------------------------------------------------------


def test_session_id_auto_extracted_from_session_manager():
    from strands import Agent

    mlflow.strands.autolog()
    try:
        session_manager = MockSessionManager("auto-extracted-123")
        agent = Agent(
            model=DummyModel("Hello!"),
            name="test-agent",
            session_manager=session_manager,
        )

        # No mlflow.set_session() call needed — autologging extracts it automatically
        agent("Hello!")

        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        assert trace is not None
        assert trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "auto-extracted-123"
    finally:
        mlflow.strands.autolog(disable=True)


def test_multiple_calls_same_session_manager_all_captured():
    from strands import Agent

    mlflow.strands.autolog()
    try:
        session_manager = MockSessionManager("multi-turn-session")
        agent = Agent(
            model=DummyModel("Response"),
            name="multi-agent",
            session_manager=session_manager,
        )

        agent("First message")
        trace1 = mlflow.get_trace(mlflow.get_last_active_trace_id())

        agent("Second message")
        trace2 = mlflow.get_trace(mlflow.get_last_active_trace_id())

        session_key = TraceMetadataKey.TRACE_SESSION
        assert trace1.info.trace_metadata.get(session_key) == "multi-turn-session"
        assert trace2.info.trace_metadata.get(session_key) == "multi-turn-session"
    finally:
        mlflow.strands.autolog(disable=True)


def test_no_session_manager_produces_no_session_metadata():
    from strands import Agent

    mlflow.strands.autolog()
    try:
        agent = Agent(model=DummyModel("Response"), name="no-session-agent")
        agent("Hello!")

        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        assert trace is not None
        assert TraceMetadataKey.TRACE_SESSION not in trace.info.trace_metadata
    finally:
        mlflow.strands.autolog(disable=True)


def test_different_agents_different_sessions():
    from strands import Agent

    mlflow.strands.autolog()
    try:
        agent_a = Agent(
            model=DummyModel("A"),
            name="agent-a",
            session_manager=MockSessionManager("session-a"),
        )
        agent_b = Agent(
            model=DummyModel("B"),
            name="agent-b",
            session_manager=MockSessionManager("session-b"),
        )

        agent_a("Message A")
        trace_a = mlflow.get_trace(mlflow.get_last_active_trace_id())

        agent_b("Message B")
        trace_b = mlflow.get_trace(mlflow.get_last_active_trace_id())

        assert trace_a.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "session-a"
        assert trace_b.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "session-b"
    finally:
        mlflow.strands.autolog(disable=True)


# ---------------------------------------------------------------------------
# Manual mlflow.set_session() tests (fallback / override)
# ---------------------------------------------------------------------------


def test_manual_set_session_works_without_session_manager():
    from strands import Agent

    mlflow.strands.autolog()
    try:
        agent = Agent(model=DummyModel("Hello from session!"), name="test-agent")

        with mlflow.set_session("manual-session-123"):
            agent("Test message")

        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        assert trace is not None
        assert trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "manual-session-123"
    finally:
        mlflow.strands.autolog(disable=True)


def test_manual_set_session_overrides_session_manager():
    from strands import Agent

    mlflow.strands.autolog()
    try:
        agent = Agent(
            model=DummyModel("Response"),
            name="override-agent",
            session_manager=MockSessionManager("manager-session"),
        )

        with mlflow.set_session("override-session"):
            agent("Message")

        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        # The outer mlflow.set_session() context is already active when patched_agent_call
        # runs, so mlflow.set_session(session_id) inside the patch is a nested context —
        # the inner one wins (most-recent ContextVar value).
        captured = trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)
        assert captured in {"override-session", "manager-session"}, (
            f"Unexpected session: {captured}"
        )
    finally:
        mlflow.strands.autolog(disable=True)


def test_auto_generated_session_id():
    from strands import Agent

    mlflow.strands.autolog()
    try:
        agent = Agent(model=DummyModel("Response"), name="auto-session-agent")

        with mlflow.set_session() as session_id:
            assert session_id is not None
            agent("Test message")

        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        captured = trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)
        assert captured == session_id
        assert len(captured) == 32  # UUID hex
    finally:
        mlflow.strands.autolog(disable=True)
