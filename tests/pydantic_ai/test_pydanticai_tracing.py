import importlib.metadata
from unittest.mock import patch

import pytest
from packaging.version import Version
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.usage import Usage

import mlflow
import mlflow.pydantic_ai  # ensure the integration module is importable
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

from tests.tracing.helper import get_traces

PYDANTIC_AI_VERSION = Version(importlib.metadata.version("pydantic_ai"))
# Usage was deprecated in favor of RequestUsage in 0.7.3
IS_USAGE_DEPRECATED = PYDANTIC_AI_VERSION >= Version("0.7.3")

_FINAL_ANSWER_WITHOUT_TOOL = "Paris"
_FINAL_ANSWER_WITH_TOOL = "winner"


def _make_dummy_response_without_tool():
    # Usage was deprecated in favor of RequestUsage in 0.7.3
    if IS_USAGE_DEPRECATED:
        from pydantic_ai.usage import RequestUsage

    parts = [TextPart(content=_FINAL_ANSWER_WITHOUT_TOOL)]
    if IS_USAGE_DEPRECATED:
        usage = RequestUsage(input_tokens=1, output_tokens=1)
    else:
        usage = Usage(requests=1, request_tokens=1, response_tokens=1, total_tokens=2)
    if PYDANTIC_AI_VERSION >= Version("0.2.0"):
        return ModelResponse(parts=parts, usage=usage)
    else:
        resp = ModelResponse(parts=parts)
        return resp, usage


def _make_dummy_response_with_tool():
    # Usage was deprecated in favor of RequestUsage in 0.7.3
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
        sequence = [
            call_resp,
            final_resp,
        ]
        return sequence, final_resp

    else:
        call_resp = ModelResponse(parts=call_parts)
        final_resp = ModelResponse(parts=final_parts)
        sequence = [
            (call_resp, usage_call),
            (final_resp, usage_final),
        ]

        return sequence, (final_resp, usage_final)


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


def test_agent_run_sync_enable_disable_autolog(simple_agent):
    dummy = _make_dummy_response_without_tool()

    async def request(self, *args, **kwargs):
        return dummy

    with patch("pydantic_ai.models.instrumented.InstrumentedModel.request", new=request):
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

    assert span2.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 1,
        TokenUsageKey.OUTPUT_TOKENS: 1,
        TokenUsageKey.TOTAL_TOKENS: 2,
    }

    assert traces[0].info.token_usage == {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
    }

    with patch("pydantic_ai.models.instrumented.InstrumentedModel.request", new=request):
        mlflow.pydantic_ai.autolog(disable=True)
        simple_agent.run_sync("France")
    assert len(get_traces()) == 1


@pytest.mark.asyncio
async def test_agent_run_enable_disable_autolog(simple_agent):
    dummy = _make_dummy_response_without_tool()

    async def request(self, *args, **kwargs):
        return dummy

    with patch("pydantic_ai.models.instrumented.InstrumentedModel.request", new=request):
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

    assert span1.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 1,
        TokenUsageKey.OUTPUT_TOKENS: 1,
        TokenUsageKey.TOTAL_TOKENS: 2,
    }

    assert traces[0].info.token_usage == {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
    }


def test_agent_run_sync_enable_disable_autolog_with_tool(agent_with_tool):
    sequence, resp = _make_dummy_response_with_tool()

    async def request(self, *args, **kwargs):
        if sequence:
            return sequence.pop(0)
        return resp

    with patch("pydantic_ai.models.instrumented.InstrumentedModel.request", new=request):
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
    assert span2.name == "InstrumentedModel.request_1"
    assert span2.span_type == SpanType.LLM
    assert span2.parent_id == spans[1].span_id

    span3 = spans[3]
    assert span3.span_type == SpanType.TOOL
    assert span3.parent_id == spans[1].span_id

    span4 = spans[4]
    assert span4.name == "InstrumentedModel.request_2"
    assert span4.span_type == SpanType.LLM
    assert span4.parent_id == spans[1].span_id

    assert span2.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 10,
        TokenUsageKey.OUTPUT_TOKENS: 20,
        TokenUsageKey.TOTAL_TOKENS: 30,
    }

    assert span4.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 100,
        TokenUsageKey.OUTPUT_TOKENS: 200,
        TokenUsageKey.TOTAL_TOKENS: 300,
    }

    assert traces[0].info.token_usage == {
        "input_tokens": 110,
        "output_tokens": 220,
        "total_tokens": 330,
    }


@pytest.mark.asyncio
async def test_agent_run_enable_disable_autolog_with_tool(agent_with_tool):
    sequence, resp = _make_dummy_response_with_tool()

    async def request(self, *args, **kwargs):
        if sequence:
            return sequence.pop(0)
        return resp

    with patch("pydantic_ai.models.instrumented.InstrumentedModel.request", new=request):
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
    assert span1.name == "InstrumentedModel.request_1"
    assert span1.span_type == SpanType.LLM
    assert span1.parent_id == spans[0].span_id

    span2 = spans[2]
    assert span2.span_type == SpanType.TOOL
    assert span2.parent_id == spans[0].span_id

    span3 = spans[3]
    assert span3.name == "InstrumentedModel.request_2"
    assert span3.span_type == SpanType.LLM
    assert span3.parent_id == spans[0].span_id

    assert span1.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 10,
        TokenUsageKey.OUTPUT_TOKENS: 20,
        TokenUsageKey.TOTAL_TOKENS: 30,
    }

    assert span3.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 100,
        TokenUsageKey.OUTPUT_TOKENS: 200,
        TokenUsageKey.TOTAL_TOKENS: 300,
    }

    assert traces[0].info.token_usage == {
        "input_tokens": 110,
        "output_tokens": 220,
        "total_tokens": 330,
    }


def test_agent_run_sync_failure(simple_agent):
    with patch("pydantic_ai.models.instrumented.InstrumentedModel.request", side_effect=Exception):
        mlflow.pydantic_ai.autolog(log_traces=True)

        with pytest.raises(Exception, match="e"):
            simple_agent.run_sync("France")

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    spans = traces[0].data.spans

    assert len(spans) == 3
    assert spans[0].name == "Agent.run_sync"
    assert spans[0].span_type == SpanType.AGENT
    assert spans[1].name == "Agent.run"
    assert spans[1].span_type == SpanType.AGENT
    assert spans[2].name == "InstrumentedModel.AsyncMock"
    assert spans[2].span_type == SpanType.LLM

    with patch("pydantic_ai.models.instrumented.InstrumentedModel.request", side_effect=Exception):
        mlflow.pydantic_ai.autolog(disable=True)

        with pytest.raises(Exception):  # noqa
            simple_agent.run_sync("France")

    traces = get_traces()
    assert len(traces) == 1
