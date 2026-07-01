import sys
from unittest.mock import MagicMock, patch

import agno
import pytest
from agno.agent import Agent
from agno.exceptions import ModelProviderError
from agno.models.anthropic import Claude
from agno.tools.function import Function, FunctionCall
from anthropic.types import Message, TextBlock, Usage
from packaging.version import Version

import mlflow
import mlflow.agno
from mlflow.entities import SpanType
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import TokenUsageKey

from tests.tracing.helper import get_traces, purge_traces

AGNO_VERSION = Version(getattr(agno, "__version__", "1.0.0"))
IS_AGNO_V2 = AGNO_VERSION >= Version("2.0.0")
# In agno >= 2.3.14, errors are caught internally and returned as error status
# instead of being raised as ModelProviderError
AGNO_CATCHES_ERRORS = AGNO_VERSION >= Version("2.3.14")


def get_v2_autolog_module():
    from mlflow.agno.autolog_v2 import _is_agno_v2  # noqa: F401

    return sys.modules["mlflow.agno.autolog_v2"]


def _create_message(content):
    return Message(
        id="1",
        model="claude-sonnet-4-20250514",
        content=[TextBlock(text=content, type="text")],
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=5, output_tokens=7, total_tokens=12),
    )


@pytest.fixture
def simple_agent():
    return Agent(
        model=Claude(id="claude-sonnet-4-20250514"),
        instructions="Be concise.",
        markdown=True,
    )


@pytest.mark.skipif(IS_AGNO_V2, reason="Test uses V1 patching behavior")
def test_run_simple_autolog(simple_agent):
    mlflow.agno.autolog()

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_message("Paris")
    with patch.object(Claude, "get_client", return_value=mock_client):
        resp = simple_agent.run("Capital of France?")
    assert resp.content == "Paris"

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 5,
        TokenUsageKey.OUTPUT_TOKENS: 7,
        TokenUsageKey.TOTAL_TOKENS: 12,
    }
    spans = traces[0].data.spans
    assert len(spans) == 2
    assert spans[0].span_type == SpanType.AGENT
    assert spans[0].name == "Agent.run"
    assert spans[0].inputs == {"message": "Capital of France?"}
    assert spans[0].outputs["content"] == "Paris"
    assert spans[1].span_type == SpanType.LLM
    assert spans[1].name == "Claude.invoke"
    assert spans[1].inputs["messages"][-1]["content"] == "Capital of France?"
    assert spans[1].outputs["content"][0]["text"] == "Paris"
    assert spans[1].model_name == "claude-sonnet-4-20250514"

    purge_traces()

    mlflow.agno.autolog(disable=True)
    with patch.object(Claude, "get_client", return_value=mock_client):
        simple_agent.run("Again?")
    assert get_traces() == []


@pytest.mark.skipif(IS_AGNO_V2, reason="Test uses V1 patching behavior")
def test_run_failure_tracing(simple_agent):
    mlflow.agno.autolog()

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("bang")
    with patch.object(Claude, "get_client", return_value=mock_client):
        with pytest.raises(ModelProviderError, match="bang"):
            simple_agent.run("fail")

    trace = get_traces()[0]
    assert trace.info.status == "ERROR"
    assert trace.info.token_usage is None
    spans = trace.data.spans
    assert spans[0].name == "Agent.run"
    assert spans[1].name == "Claude.invoke"
    assert spans[1].status.status_code == SpanStatusCode.ERROR
    assert spans[1].status.description == "ModelProviderError: bang"


@pytest.mark.skipif(IS_AGNO_V2, reason="Test uses V1 patching behavior")
@pytest.mark.asyncio
async def test_arun_simple_autolog(simple_agent):
    mlflow.agno.autolog()

    async def _mock_create(*args, **kwargs):
        return _create_message("Paris")

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = _mock_create
    with patch.object(Claude, "get_async_client", return_value=mock_client):
        resp = await simple_agent.arun("Capital of France?")

    assert resp.content == "Paris"

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 5,
        TokenUsageKey.OUTPUT_TOKENS: 7,
        TokenUsageKey.TOTAL_TOKENS: 12,
    }
    spans = traces[0].data.spans
    assert len(spans) == 2
    assert spans[0].span_type == SpanType.AGENT
    assert spans[0].name == "Agent.arun"
    assert spans[0].inputs == {"message": "Capital of France?"}
    assert spans[0].outputs["content"] == "Paris"
    assert spans[1].span_type == SpanType.LLM
    assert spans[1].name == "Claude.ainvoke"
    assert spans[1].inputs["messages"][-1]["content"] == "Capital of France?"
    assert spans[1].outputs["content"][0]["text"] == "Paris"
    assert spans[1].model_name == "claude-sonnet-4-20250514"


@pytest.mark.skipif(IS_AGNO_V2, reason="Test uses V1 patching behavior")
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
async def test_failure_tracing(simple_agent, is_async):
    mlflow.agno.autolog()

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("bang")
    mock_method = "get_async_client" if is_async else "get_client"
    with patch.object(Claude, mock_method, return_value=mock_client):
        with pytest.raises(ModelProviderError, match="bang"):  # noqa: PT012
            if is_async:
                await simple_agent.arun("fail")
            else:
                simple_agent.run("fail")

    trace = get_traces()[0]
    assert trace.info.status == "ERROR"
    assert trace.info.token_usage is None
    spans = trace.data.spans
    assert spans[0].name == "Agent.run" if not is_async else "Agent.arun"
    assert spans[1].name == "Claude.invoke" if not is_async else "Claude.ainvoke"
    assert spans[1].status.status_code == SpanStatusCode.ERROR
    assert spans[1].status.description == "ModelProviderError: bang"


@pytest.mark.skipif(IS_AGNO_V2, reason="Test uses V1 patching behavior")
def test_function_execute_tracing():
    def dummy(x):
        return x + 1

    fc = FunctionCall(function=Function.from_callable(dummy, name="dummy"), arguments={"x": 1})

    mlflow.agno.autolog(log_traces=True)
    result = fc.execute()
    assert result.result == 2

    spans = get_traces()[0].data.spans
    assert len(spans) == 1
    span = spans[0]
    assert span.span_type == SpanType.TOOL
    assert span.name == "dummy"
    assert span.inputs == {"x": 1}
    assert span.attributes["entrypoint"] is not None
    assert span.outputs["result"] == 2


@pytest.mark.skipif(IS_AGNO_V2, reason="Test uses V1 patching behavior")
@pytest.mark.asyncio
async def test_function_aexecute_tracing():
    async def dummy(x):
        return x + 1

    fc = FunctionCall(function=Function.from_callable(dummy, name="dummy"), arguments={"x": 1})

    mlflow.agno.autolog(log_traces=True)
    result = await fc.aexecute()
    assert result.result == 2

    spans = get_traces()[0].data.spans
    assert len(spans) == 1
    span = spans[0]
    assert span.span_type == SpanType.TOOL
    assert span.name == "dummy"
    assert span.inputs == {"x": 1}
    assert span.attributes["entrypoint"] is not None
    assert span.outputs["result"] == 2


@pytest.mark.skipif(IS_AGNO_V2, reason="Test uses V1 patching behavior")
def test_function_execute_failure_tracing():
    from agno.exceptions import AgentRunException

    def boom(x):
        raise AgentRunException("bad")

    fc = FunctionCall(function=Function.from_callable(boom, name="boom"), arguments={"x": 1})

    mlflow.agno.autolog(log_traces=True)
    with pytest.raises(AgentRunException, match="bad"):
        fc.execute()

    trace = get_traces()[0]
    assert trace.info.status == "ERROR"
    span = trace.data.spans[0]
    assert span.span_type == SpanType.TOOL
    assert span.status.status_code == SpanStatusCode.ERROR
    assert span.inputs == {"x": 1}
    assert span.outputs is None


@pytest.mark.skipif(IS_AGNO_V2, reason="Test uses V1 patching behavior")
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
async def test_agno_and_anthropic_autolog_single_trace(simple_agent, is_async):
    mlflow.agno.autolog()
    mlflow.anthropic.autolog()

    client = "AsyncAPIClient" if is_async else "SyncAPIClient"
    with patch(f"anthropic._base_client.{client}.post", return_value=_create_message("Paris")):
        if is_async:
            await simple_agent.arun("hi")
        else:
            simple_agent.run("hi")

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert spans[0].span_type == SpanType.AGENT
    assert spans[0].name == "Agent.arun" if is_async else "Agent.run"
    assert spans[1].span_type == SpanType.LLM
    assert spans[1].name == "Claude.ainvoke" if is_async else "Claude.invoke"
    assert spans[2].span_type == SpanType.CHAT_MODEL
    assert spans[2].name == "AsyncMessages.create" if is_async else "Messages.create"


@pytest.mark.skipif(not IS_AGNO_V2, reason="Test requires V2 functionality")
def test_v2_autolog_setup_teardown():
    autolog_module = get_v2_autolog_module()
    original_instrumentor = autolog_module._agno_instrumentor

    try:
        autolog_module._agno_instrumentor = None

        with patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"):
            mlflow.agno.autolog(log_traces=True)
            assert autolog_module._agno_instrumentor is not None

            mlflow.agno.autolog(log_traces=False)
    finally:
        autolog_module._agno_instrumentor = original_instrumentor


@pytest.mark.skipif(not IS_AGNO_V2, reason="Test requires V2 functionality")
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
async def test_v2_creates_otel_spans(simple_agent, is_async):
    try:
        mlflow.agno.autolog(log_traces=True)

        mock_client = MagicMock()
        if is_async:

            async def _mock_create(*args, **kwargs):
                return _create_message("Paris")

            mock_client.messages.create.side_effect = _mock_create
        else:
            mock_client.messages.create.return_value = _create_message("Paris")

        mock_method = "get_async_client" if is_async else "get_client"
        with patch.object(Claude, mock_method, return_value=mock_client):
            if is_async:
                resp = await simple_agent.arun("Capital of France?")
            else:
                resp = simple_agent.run("Capital of France?")

        assert resp.content == "Paris"

        # Agno spans are routed through MLflow's own tracer provider, so they are captured as an
        # MLflow trace rather than emitted to the global OpenTelemetry provider.
        traces = get_traces()
        assert len(traces) == 1
        assert len(traces[0].data.spans) > 0
    finally:
        mlflow.agno.autolog(disable=True)


@pytest.mark.skipif(not IS_AGNO_V2, reason="Test requires V2 functionality")
def test_v2_failure_creates_spans(simple_agent):
    try:
        mlflow.agno.autolog(log_traces=True)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("bang")
        with patch.object(Claude, "get_client", return_value=mock_client):
            if AGNO_CATCHES_ERRORS:
                # In agno >= 2.3.14, errors are caught internally and returned as error status
                from agno.run import RunStatus

                result = simple_agent.run("fail")
                assert result.status == RunStatus.error
                assert "bang" in result.content
            else:
                # In agno < 2.3.14, errors are raised as ModelProviderError
                with pytest.raises(ModelProviderError, match="bang"):
                    simple_agent.run("fail")

        traces = get_traces()
        assert len(traces) == 1
        spans = traces[0].data.spans
        assert len(spans) > 0
        if not AGNO_CATCHES_ERRORS:
            # Error spans are only created when exceptions propagate
            error_spans = [s for s in spans if s.status.status_code == SpanStatusCode.ERROR]
            assert len(error_spans) > 0
    finally:
        mlflow.agno.autolog(disable=True)


@pytest.mark.skipif(not IS_AGNO_V2, reason="Test requires V2 functionality")
def test_v2_spans_nest_under_manual_mlflow_span(simple_agent):
    # Agno's OpenInference spans must nest under a manually-created mlflow.start_span() span
    # (one combined trace) rather than starting their own disconnected trace.
    try:
        mlflow.agno.autolog(log_traces=True)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _create_message("Paris")
        with patch.object(Claude, "get_client", return_value=mock_client):
            with mlflow.start_span(name="outer"):
                resp = simple_agent.run("Capital of France?")

        assert resp.content == "Paris"

        traces = get_traces()
        assert len(traces) == 1
        spans = traces[0].data.spans

        # The manual span is the single root, and there are Agno spans on top of it.
        roots = [s for s in spans if s.parent_id is None]
        assert len(roots) == 1
        assert roots[0].name == "outer"
        assert len(spans) > 1

        # Every span (including the Agno ones) descends from the manual "outer" root.
        by_id = {s.span_id: s for s in spans}
        for span in spans:
            current = span
            while current.parent_id is not None:
                current = by_id[current.parent_id]
            assert current.name == "outer"
    finally:
        mlflow.agno.autolog(disable=True)
