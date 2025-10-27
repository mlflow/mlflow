from unittest.mock import MagicMock, patch

import pytest
from agno.agent import Agent
from agno.exceptions import ModelProviderError
from agno.models.anthropic import Claude
from agno.tools.function import Function, FunctionCall
from anthropic.types import Message, TextBlock, Usage

import mlflow
import mlflow.agno
from mlflow.entities import SpanType
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import TokenUsageKey

from tests.tracing.helper import get_traces, purge_traces


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
    # Agno add system message to the input messages, so validate the last message
    assert spans[1].inputs["messages"][-1]["content"] == "Capital of France?"
    assert spans[1].outputs["content"][0]["text"] == "Paris"

    purge_traces()

    mlflow.agno.autolog(disable=True)
    with patch.object(Claude, "get_client", return_value=mock_client):
        simple_agent.run("Again?")
    assert get_traces() == []


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
    # Agno add system message to the input messages, so validate the last message
    assert spans[1].inputs["messages"][-1]["content"] == "Capital of France?"
    assert spans[1].outputs["content"][0]["text"] == "Paris"


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
