import importlib.metadata
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from packaging.version import Version
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.usage import Usage

import mlflow
import mlflow.pydantic_ai  # ensure the integration module is importable
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces

_FINAL_ANSWER_WITHOUT_TOOL = "Paris"
_FINAL_ANSWER_WITH_TOOL = "winner"

PYDANTIC_AI_VERSION = Version(importlib.metadata.version("pydantic_ai"))
# Usage was deprecated in favor of RequestUsage in 0.7.3
IS_USAGE_DEPRECATED = PYDANTIC_AI_VERSION >= Version("0.7.3")
# run_stream_sync was added in pydantic-ai 1.10.0
HAS_RUN_STREAM_SYNC = hasattr(Agent, "run_stream_sync")
# Streaming tests require pydantic-ai >= 1.0.0 due to API changes
HAS_STABLE_STREAMING_API = PYDANTIC_AI_VERSION >= Version("1.0.0")


def _make_dummy_response_without_tool():
    if IS_USAGE_DEPRECATED:
        from pydantic_ai.usage import RequestUsage

    parts = [TextPart(content=_FINAL_ANSWER_WITHOUT_TOOL)]
    resp = ModelResponse(parts=parts)
    if IS_USAGE_DEPRECATED:
        usage = RequestUsage(input_tokens=1, output_tokens=1)
    else:
        usage = Usage(requests=1, request_tokens=1, response_tokens=1, total_tokens=2)

    if PYDANTIC_AI_VERSION >= Version("0.2.0"):
        return ModelResponse(parts=parts, usage=usage)
    else:
        return resp, usage


def _make_dummy_response_with_tool():
    if IS_USAGE_DEPRECATED:
        from pydantic_ai.usage import RequestUsage

    call_parts = [ToolCallPart(tool_name="roulette_wheel", args={"square": 18})]
    final_parts = [TextPart(content=_FINAL_ANSWER_WITH_TOOL)]

    if IS_USAGE_DEPRECATED:
        usage_call = RequestUsage(input_tokens=10, output_tokens=20)
        usage_final = RequestUsage(input_tokens=100, output_tokens=200)
    else:
        usage_call = Usage(requests=0, request_tokens=10, response_tokens=20, total_tokens=30)
        usage_final = Usage(requests=1, request_tokens=100, response_tokens=200, total_tokens=300)

    if PYDANTIC_AI_VERSION >= Version("0.2.0"):
        call_resp = ModelResponse(parts=call_parts, usage=usage_call)
        final_resp = ModelResponse(parts=final_parts, usage=usage_final)
        yield call_resp
        yield final_resp

    else:
        call_resp = ModelResponse(parts=call_parts)
        final_resp = ModelResponse(parts=final_parts)
        yield call_resp, usage_call
        yield final_resp, usage_final


def _make_streaming_response_without_tool(input_tokens=10, output_tokens=5):
    if IS_USAGE_DEPRECATED:
        from pydantic_ai.usage import RequestUsage

        usage = RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    else:
        usage = Usage(
            requests=1,
            request_tokens=input_tokens,
            response_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    return ModelResponse(parts=[TextPart(content=_FINAL_ANSWER_WITHOUT_TOOL)], usage=usage), usage


def _make_streaming_response_with_tool():
    if IS_USAGE_DEPRECATED:
        from pydantic_ai.usage import RequestUsage

        usage_call = RequestUsage(input_tokens=10, output_tokens=20)
        usage_final = RequestUsage(input_tokens=100, output_tokens=200)
    else:
        usage_call = Usage(requests=0, request_tokens=10, response_tokens=20, total_tokens=30)
        usage_final = Usage(requests=1, request_tokens=100, response_tokens=200, total_tokens=300)

    call_resp = ModelResponse(
        parts=[ToolCallPart(tool_name="roulette_wheel", args={"square": 18})],
        usage=usage_call,
    )
    final_resp = ModelResponse(
        parts=[TextPart(content=_FINAL_ANSWER_WITH_TOOL)],
        usage=usage_final,
    )

    return [call_resp, final_resp]


class MockStreamedResponse:
    def __init__(self, response, usage):
        self._response = response
        self._usage = usage
        self.model_name = "openai:gpt-4o"
        self.timestamp = None

    def usage(self):
        return self._usage

    def get(self):
        return self._response

    async def __aiter__(self):
        for part in self._response.parts:
            if hasattr(part, "content"):
                for char in part.content:
                    yield char
            else:
                yield ""


@pytest.fixture(autouse=True)
def clear_autolog_state():
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    for key in AUTOLOGGING_INTEGRATIONS.keys():
        AUTOLOGGING_INTEGRATIONS[key].clear()
    mlflow.utils.import_hooks._post_import_hooks = {}


@pytest.fixture
def simple_agent():
    return Agent(
        "openai:gpt-4o",
        system_prompt="Tell me the capital of {{input}}.",
        instrument=True,
    )


@pytest.fixture
def agent_with_tool():
    roulette_agent = Agent(
        "openai:gpt-4o",
        system_prompt=(
            "Use the roulette_wheel function to see if the "
            "customer has won based on the number they provide."
        ),
        instrument=True,
        deps_type=int,
        output_type=str,
    )

    @roulette_agent.tool
    async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
        """check if the square is a winner"""
        return "winner" if square == ctx.deps else "loser"

    return roulette_agent


def test_agent_run_sync_enable_fluent_disable_autolog(simple_agent):
    dummy = _make_dummy_response_without_tool()

    async def request(self, *args, **kwargs):
        return dummy

    with patch.object(InstrumentedModel, "request", new=request):
        mlflow.pydantic_ai.autolog(log_traces=True)

        result = simple_agent.run_sync("France")
        assert result.output == _FINAL_ANSWER_WITHOUT_TOOL

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans

    assert spans[0].name == "Agent.run_sync"
    assert spans[0].span_type == SpanType.AGENT

    assert spans[1].name == "Agent.run"
    assert spans[1].span_type == SpanType.AGENT

    span2 = spans[2]
    assert span2.name == "InstrumentedModel.request"
    assert span2.span_type == SpanType.LLM
    assert span2.parent_id == spans[1].span_id

    with patch.object(InstrumentedModel, "request", new=request):
        mlflow.pydantic_ai.autolog(disable=True)
        simple_agent.run_sync("France")
    assert len(get_traces()) == 1


@pytest.mark.asyncio
async def test_agent_run_enable_fluent_disable_autolog(simple_agent):
    dummy = _make_dummy_response_without_tool()

    async def request(self, *args, **kwargs):
        return dummy

    with patch.object(InstrumentedModel, "request", new=request):
        mlflow.pydantic_ai.autolog(log_traces=True)

        result = await simple_agent.run("France")
        assert result.output == _FINAL_ANSWER_WITHOUT_TOOL

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans

    assert spans[0].name == "Agent.run"
    assert spans[0].span_type == SpanType.AGENT

    span1 = spans[1]
    assert span1.name == "InstrumentedModel.request"
    assert span1.span_type == SpanType.LLM
    assert span1.parent_id == spans[0].span_id


def test_agent_run_sync_enable_disable_fluent_autolog_with_tool(agent_with_tool):
    sequence = _make_dummy_response_with_tool()

    async def request(self, *args, **kwargs):
        return next(sequence)

    with patch.object(InstrumentedModel, "request", new=request):
        mlflow.pydantic_ai.autolog(log_traces=True)

        result = agent_with_tool.run_sync("Put my money on square eighteen", deps=18)
        assert result.output == _FINAL_ANSWER_WITH_TOOL

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans

    assert len(spans) == 5

    assert spans[0].name == "Agent.run_sync"
    assert spans[0].span_type == SpanType.AGENT

    assert spans[1].name == "Agent.run"
    assert spans[1].span_type == SpanType.AGENT

    span2 = spans[2]
    assert span2.name == "InstrumentedModel.request"
    assert span2.span_type == SpanType.LLM
    assert span2.parent_id == spans[1].span_id

    span3 = spans[3]
    assert span3.span_type == SpanType.TOOL
    assert span3.parent_id == spans[1].span_id

    span4 = spans[4]
    assert span4.name == "InstrumentedModel.request"
    assert span4.span_type == SpanType.LLM
    assert span4.parent_id == spans[1].span_id


@pytest.mark.asyncio
async def test_agent_run_enable_disable_fluent_autolog_with_tool(agent_with_tool):
    sequence = _make_dummy_response_with_tool()

    async def request(self, *args, **kwargs):
        return next(sequence)

    with patch.object(InstrumentedModel, "request", new=request):
        mlflow.pydantic_ai.autolog(log_traces=True)

        result = await agent_with_tool.run("Put my money on square eighteen", deps=18)
        assert result.output == _FINAL_ANSWER_WITH_TOOL

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans

    assert len(spans) == 4

    assert spans[0].name == "Agent.run"
    assert spans[0].span_type == SpanType.AGENT

    span1 = spans[1]
    assert span1.name == "InstrumentedModel.request"
    assert span1.span_type == SpanType.LLM
    assert span1.parent_id == spans[0].span_id

    span2 = spans[2]
    assert span2.span_type == SpanType.TOOL
    assert span2.parent_id == spans[0].span_id

    span3 = spans[3]
    assert span3.name == "InstrumentedModel.request"
    assert span3.span_type == SpanType.LLM
    assert span3.parent_id == spans[0].span_id


@pytest.mark.skipif(
    not HAS_STABLE_STREAMING_API, reason="Streaming API stabilized in pydantic-ai 1.0.0"
)
@pytest.mark.asyncio
async def test_agent_run_stream_creates_trace(simple_agent):
    response, usage = _make_streaming_response_without_tool(input_tokens=10, output_tokens=5)

    @asynccontextmanager
    async def request_stream(self, *args, **kwargs):
        yield MockStreamedResponse(response, usage)

    with patch.object(InstrumentedModel, "request_stream", new=request_stream):
        mlflow.pydantic_ai.autolog(log_traces=True)

        async with simple_agent.run_stream("France") as result:
            output = await result.get_output()
            assert output == _FINAL_ANSWER_WITHOUT_TOOL

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans

    assert len(spans) == 2

    assert spans[0].name == "Agent.run_stream"
    assert spans[0].span_type == SpanType.AGENT

    assert spans[1].name == "InstrumentedModel.request_stream"
    assert spans[1].span_type == SpanType.LLM
    assert spans[1].parent_id == spans[0].span_id

    usage_attr = spans[0].attributes.get(SpanAttributeKey.CHAT_USAGE)
    assert usage_attr is not None
    assert usage_attr.get("input_tokens") == 10
    assert usage_attr.get("output_tokens") == 5
    assert usage_attr.get("total_tokens") == 15


@pytest.mark.skipif(
    not HAS_STABLE_STREAMING_API, reason="Streaming API stabilized in pydantic-ai 1.0.0"
)
@pytest.mark.skipif(not HAS_RUN_STREAM_SYNC, reason="run_stream_sync added in pydantic-ai 1.10.0")
def test_agent_run_stream_sync_creates_trace(simple_agent):
    response, usage = _make_streaming_response_without_tool(input_tokens=10, output_tokens=5)

    @asynccontextmanager
    async def request_stream(self, *args, **kwargs):
        yield MockStreamedResponse(response, usage)

    with patch.object(InstrumentedModel, "request_stream", new=request_stream):
        mlflow.pydantic_ai.autolog(log_traces=True)

        result = simple_agent.run_stream_sync("France")
        output = ""
        for text in result.stream_text():
            output += text

    assert output == _FINAL_ANSWER_WITHOUT_TOOL

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans

    assert len(spans) == 2

    assert spans[0].name == "Agent.run_stream_sync"
    assert spans[0].span_type == SpanType.AGENT
    assert spans[0].inputs is not None
    assert "user_prompt" in spans[0].inputs
    assert spans[0].outputs is not None

    assert spans[1].name == "InstrumentedModel.request_stream"
    assert spans[1].span_type == SpanType.LLM
    assert spans[1].parent_id == spans[0].span_id

    usage_attr = spans[0].attributes.get(SpanAttributeKey.CHAT_USAGE)
    assert usage_attr is not None
    assert usage_attr.get("input_tokens") == 10
    assert usage_attr.get("output_tokens") == 5
    assert usage_attr.get("total_tokens") == 15


@pytest.mark.skipif(
    not HAS_STABLE_STREAMING_API, reason="Streaming API stabilized in pydantic-ai 1.0.0"
)
@pytest.mark.asyncio
async def test_agent_run_stream_with_tool(agent_with_tool):
    sequence = _make_streaming_response_with_tool()

    @asynccontextmanager
    async def request_stream(self, *args, **kwargs):
        if sequence:
            resp = sequence.pop(0)
            yield MockStreamedResponse(resp, resp.usage)
        else:
            resp = sequence[-1]
            yield MockStreamedResponse(resp, resp.usage)

    with patch.object(InstrumentedModel, "request_stream", new=request_stream):
        mlflow.pydantic_ai.autolog(log_traces=True)

        async with agent_with_tool.run_stream("Put my money on square eighteen", deps=18) as result:
            output = await result.get_output()
            assert output == _FINAL_ANSWER_WITH_TOOL

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans

    assert len(spans) == 4

    assert spans[0].name == "Agent.run_stream"
    assert spans[0].span_type == SpanType.AGENT

    assert spans[1].name == "InstrumentedModel.request_stream"
    assert spans[1].span_type == SpanType.LLM
    assert spans[1].parent_id == spans[0].span_id

    assert spans[2].span_type == SpanType.TOOL
    assert spans[2].name == "ToolManager.handle_call"
    assert spans[2].parent_id == spans[0].span_id

    assert spans[3].name == "InstrumentedModel.request_stream"
    assert spans[3].span_type == SpanType.LLM
    assert spans[3].parent_id == spans[0].span_id


@pytest.mark.skipif(
    not HAS_STABLE_STREAMING_API, reason="Streaming API stabilized in pydantic-ai 1.0.0"
)
@pytest.mark.skipif(not HAS_RUN_STREAM_SYNC, reason="run_stream_sync added in pydantic-ai 1.10.0")
def test_agent_run_stream_sync_with_tool(agent_with_tool):
    sequence = _make_streaming_response_with_tool()

    @asynccontextmanager
    async def request_stream(self, *args, **kwargs):
        if sequence:
            resp = sequence.pop(0)
            yield MockStreamedResponse(resp, resp.usage)
        else:
            resp = sequence[-1]
            yield MockStreamedResponse(resp, resp.usage)

    with patch.object(InstrumentedModel, "request_stream", new=request_stream):
        mlflow.pydantic_ai.autolog(log_traces=True)

        result = agent_with_tool.run_stream_sync("Put my money on square eighteen", deps=18)
        output = ""
        for text in result.stream_text():
            output += text

    assert output == _FINAL_ANSWER_WITH_TOOL

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans

    assert len(spans) == 4

    assert spans[0].name == "Agent.run_stream_sync"
    assert spans[0].span_type == SpanType.AGENT
    assert spans[0].inputs is not None
    assert "user_prompt" in spans[0].inputs

    assert spans[1].name == "InstrumentedModel.request_stream"
    assert spans[1].span_type == SpanType.LLM
    assert spans[1].parent_id == spans[0].span_id

    assert spans[2].span_type == SpanType.TOOL
    assert spans[2].name == "ToolManager.handle_call"
    assert spans[2].parent_id == spans[0].span_id

    assert spans[3].name == "InstrumentedModel.request_stream"
    assert spans[3].span_type == SpanType.LLM
    assert spans[3].parent_id == spans[0].span_id
