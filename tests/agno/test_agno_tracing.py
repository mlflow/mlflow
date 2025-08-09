import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.function import Function, FunctionCall
from anthropic.resources import Messages
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, Usage

import mlflow
import mlflow.agno
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces


def _safe_resp(content, *, calls=None, metrics=None):
    return SimpleNamespace(
        content=content,
        metrics=metrics or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        tool_calls=calls or [],
        tool_executions=[],
        thinking="",
        redacted_thinking="",
        citations=SimpleNamespace(urls=[]),
        audio=[],
        image=[],
        created_at=[],
    )


@pytest.fixture
def simple_agent():
    return Agent(
        model=Claude(id="claude-sonnet-4-20250514"),
        instructions="Be concise.",
        markdown=True,
    )


def test_run_simple_autolog(simple_agent):
    with patch.object(Claude, "response", lambda self, messages, **kw: _safe_resp("Paris")):
        mlflow.agno.autolog(log_traces=True)
        resp = simple_agent.run("Capital of France?")
    assert resp.content == "Paris"

    spans = [s.span_type for s in get_traces()[0].data.spans]
    assert spans == [SpanType.AGENT]

    with patch.object(Claude, "response", lambda self, messages, **kw: _safe_resp("Paris")):
        mlflow.agno.autolog(disable=True)
        simple_agent.run("Again?")
    assert len(get_traces()) == 1


def test_run_failure_tracing(simple_agent):
    def _boom(self, messages, **kw):
        raise RuntimeError("bang")

    with patch.object(Claude, "response", new=_boom):
        mlflow.agno.autolog(log_traces=True)
        with pytest.raises(RuntimeError, match="bang"):
            simple_agent.run("fail")

    trace = get_traces()[0]
    assert trace.info.status == "ERROR"
    spans = [s.span_type for s in trace.data.spans]
    assert spans == [SpanType.AGENT]


@pytest.mark.asyncio
async def test_arun_simple_autolog(simple_agent):
    async def _resp(self, messages, **kw):
        return _safe_resp("Paris")

    with patch.object(Claude, "aresponse", _resp):
        mlflow.agno.autolog(log_traces=True)
        resp = await simple_agent.arun("Capital of France?")
    assert resp.content == "Paris"

    spans = [s.span_type for s in get_traces()[0].data.spans]
    assert spans == [SpanType.AGENT]


@pytest.mark.asyncio
async def test_arun_failure_tracing(simple_agent):
    async def _boom(self, messages, **kw):
        raise RuntimeError("bang")

    with patch.object(Claude, "aresponse", new=_boom):
        mlflow.agno.autolog(log_traces=True)
        with pytest.raises(RuntimeError, match="bang"):
            await simple_agent.arun("fail")

    trace = get_traces()[0]
    assert trace.info.status == "ERROR"
    spans = [s.span_type for s in trace.data.spans]
    assert spans == [SpanType.AGENT]


def test_function_execute_tracing():
    def dummy(x):
        return x + 1

    fc = FunctionCall(function=Function.from_callable(dummy, name="dummy"), arguments={"x": 1})

    mlflow.agno.autolog(log_traces=True)
    fc.execute()

    spans = get_traces()[0].data.spans
    assert len(spans) == 1
    span = spans[0]
    assert span.span_type == SpanType.TOOL
    assert span.name == "dummy"


@pytest.mark.asyncio
async def test_function_aexecute_tracing():
    async def dummy(x):
        return x + 1

    fc = FunctionCall(function=Function.from_callable(dummy, name="dummy"), arguments={"x": 1})

    mlflow.agno.autolog(log_traces=True)
    await fc.aexecute()

    spans = get_traces()[0].data.spans
    assert len(spans) == 1
    span = spans[0]
    assert span.span_type == SpanType.TOOL
    assert span.name == "dummy"


def test_agent_run_with_function_span(simple_agent):
    def dummy_tool():
        return "ok"

    func = Function.from_callable(dummy_tool, name="dummy_tool")

    def patched(self, messages, **kw):
        FunctionCall(function=func).execute()
        return _safe_resp("done")

    with patch.object(Claude, "response", patched):
        mlflow.agno.autolog(log_traces=True)
        simple_agent.run("hi")

    spans = get_traces()[0].data.spans
    assert [s.span_type for s in spans] == [SpanType.AGENT, SpanType.TOOL]
    assert spans[1].name == "dummy_tool"


@pytest.mark.asyncio
async def test_agent_arun_with_function_span(simple_agent):
    def dummy_tool():
        return "ok"

    func = Function.from_callable(dummy_tool, name="dummy_tool")

    async def patched(self, messages, **kw):
        await FunctionCall(function=func).aexecute()
        return _safe_resp("done")

    with patch.object(Claude, "aresponse", patched):
        mlflow.agno.autolog(log_traces=True)
        await simple_agent.arun("hi")

    spans = get_traces()[0].data.spans
    assert [s.span_type for s in spans] == [SpanType.AGENT, SpanType.TOOL]
    assert spans[1].name == "dummy_tool"


def test_token_usage_recorded(simple_agent):
    metrics = {"input_tokens": [5], "output_tokens": [7], "total_tokens": [12]}

    agno_autolog_module = importlib.import_module("mlflow.agno.autolog")

    expected = {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}

    with patch.object(agno_autolog_module, "_parse_usage", lambda result: expected):
        with patch.object(
            Claude, "response", lambda self, messages, **kw: _safe_resp("ok", metrics=metrics)
        ):
            mlflow.agno.autolog(log_traces=True)
            simple_agent.run("hi")

    trace = get_traces()[0]
    span = trace.data.spans[0]
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == expected
    assert trace.info.token_usage == expected


def test_token_usage_missing(simple_agent):
    with patch.object(
        Claude, "response", lambda self, messages, **kw: _safe_resp("ok", metrics=None)
    ):
        mlflow.agno.autolog(log_traces=True)
        simple_agent.run("hi")

    trace = get_traces()[0]
    span = trace.data.spans[0]
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) is None
    assert trace.info.token_usage is None


def test_autolog_disable_prevents_tool_traces():
    def dummy():
        return "x"

    fc = FunctionCall(function=Function.from_callable(dummy, name="dummy"))

    mlflow.agno.autolog(log_traces=True)
    fc.execute()
    assert len(get_traces()) == 1

    mlflow.agno.autolog(disable=True)
    FunctionCall(function=Function.from_callable(dummy, name="dummy")).execute()
    assert len(get_traces()) == 1


def test_multiple_agent_runs(simple_agent):
    with patch.object(Claude, "response", lambda self, messages, **kw: _safe_resp("A")):
        mlflow.agno.autolog(log_traces=True)
        simple_agent.run("hi")
    with patch.object(Claude, "response", lambda self, messages, **kw: _safe_resp("B")):
        simple_agent.run("hello")

    traces = get_traces()
    assert len(traces) == 2


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


@pytest.mark.asyncio
async def test_function_aexecute_failure_tracing():
    from agno.exceptions import AgentRunException

    async def boom(x):
        raise AgentRunException("bad")

    fc = FunctionCall(function=Function.from_callable(boom, name="boom"), arguments={"x": 1})

    mlflow.agno.autolog(log_traces=True)
    with pytest.raises(AgentRunException, match="bad"):
        await fc.aexecute()

    trace = get_traces()[0]
    assert trace.info.status == "ERROR"
    span = trace.data.spans[0]
    assert span.span_type == SpanType.TOOL


def test_agent_run_with_multiple_function_spans(simple_agent):
    def tool1():
        return "a"

    def tool2():
        return "b"

    func1 = Function.from_callable(tool1, name="tool1")
    func2 = Function.from_callable(tool2, name="tool2")

    def patched(self, messages, **kw):
        FunctionCall(function=func1).execute()
        FunctionCall(function=func2).execute()
        return _safe_resp("done")

    with patch.object(Claude, "response", patched):
        mlflow.agno.autolog(log_traces=True)
        simple_agent.run("hi")

    spans = get_traces()[0].data.spans
    assert [s.span_type for s in spans] == [SpanType.AGENT, SpanType.TOOL, SpanType.TOOL]
    assert [s.name for s in spans[1:]] == ["tool1", "tool2"]


@pytest.mark.asyncio
async def test_agent_arun_with_multiple_function_spans(simple_agent):
    def tool1():
        return "a"

    def tool2():
        return "b"

    func1 = Function.from_callable(tool1, name="tool1")
    func2 = Function.from_callable(tool2, name="tool2")

    async def patched(self, messages, **kw):
        await FunctionCall(function=func1).aexecute()
        await FunctionCall(function=func2).aexecute()
        return _safe_resp("done")

    with patch.object(Claude, "aresponse", patched):
        mlflow.agno.autolog(log_traces=True)
        await simple_agent.arun("hi")

    spans = get_traces()[0].data.spans
    assert [s.span_type for s in spans] == [SpanType.AGENT, SpanType.TOOL, SpanType.TOOL]
    assert [s.name for s in spans[1:]] == ["tool1", "tool2"]


def test_autolog_log_traces_false_produces_no_traces(simple_agent):
    with patch.object(Claude, "response", lambda self, messages, **kw: _safe_resp("ok")):
        mlflow.agno.autolog(log_traces=False)
        simple_agent.run("hi")

    assert get_traces() == []


def test_agno_and_anthropic_autolog_single_trace(simple_agent, monkeypatch):
    dummy = AnthropicMessage(
        id="1",
        content=[TextBlock(text="ok", type="text", citations=None)],
        model="m",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    def _create(self, *a, **k):
        return dummy

    _create.__name__ = "create"

    with patch.object(Messages, "create", _create):
        mlflow.anthropic.autolog(log_traces=True)
        mlflow.agno.autolog(log_traces=True)
        simple_agent.run("hi")

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert [s.span_type for s in spans] == [SpanType.AGENT, SpanType.CHAT_MODEL]
    assert spans[1].name == "Messages.create"


@pytest.mark.asyncio
async def test_agno_and_anthropic_autolog_single_trace_async(simple_agent, monkeypatch):
    dummy = AnthropicMessage(
        id="1",
        content=[TextBlock(text="ok", type="text", citations=None)],
        model="m",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    def _create(self, *a, **k):
        return dummy

    _create.__name__ = "create"

    async def _aresponse(self, messages, **kwargs):
        return self.response(messages, **kwargs)

    with patch.object(Messages, "create", _create), patch.object(Claude, "aresponse", _aresponse):
        mlflow.anthropic.autolog(log_traces=True)
        mlflow.agno.autolog(log_traces=True)
        await simple_agent.arun("hi")

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert [s.span_type for s in spans] == [SpanType.AGENT, SpanType.CHAT_MODEL]
    assert spans[1].name == "Messages.create"
