from unittest.mock import patch

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.usage import Usage

import mlflow
import mlflow.pydantic_ai  # ensure the integration module is importable
from mlflow.entities import SpanType

from tests.tracing.helper import get_traces

_FINAL_ANSWER_WITHOUT_TOOL = "Paris"
_FINAL_ANSWER_WITH_TOOL = "winner"


@pytest.fixture(autouse=True)
def mock_openai_creds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "my-secret-key")


@pytest.fixture(autouse=True)
def mock_gemini_creds(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "my-secret-key")


@pytest.fixture(autouse=True)
def reset_mlflow_autolog_and_traces():
    mlflow.autolog(disable=True)
    get_traces().clear()
    yield
    mlflow.autolog(disable=True)
    get_traces().clear()


def _make_dummy_response_without_tool():
    part = TextPart(content=_FINAL_ANSWER_WITHOUT_TOOL)
    resp = ModelResponse(parts=[part])
    usage = Usage(requests=1, request_tokens=1, response_tokens=1, total_tokens=2)
    return resp, usage


def _make_dummy_response_with_tool():
    call_resp = ModelResponse(parts=[ToolCallPart(tool_name="roulette_wheel", args={"square": 18})])
    usage_call = Usage(requests=0, request_tokens=10, response_tokens=20, total_tokens=30)

    final_resp = ModelResponse(parts=[TextPart(content=_FINAL_ANSWER_WITH_TOOL)])
    usage_final = Usage(requests=1, request_tokens=100, response_tokens=200, total_tokens=300)

    sequence = [
        (call_resp, usage_call),
        (final_resp, usage_final),
    ]

    return sequence, final_resp, usage_final


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
        mlflow.autolog(log_traces=True)

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
        mlflow.autolog(disable=True)
        simple_agent.run_sync("France")
    assert len(get_traces()) == 1


@pytest.mark.asyncio
async def test_agent_run_enable_fluent_disable_autolog(simple_agent):
    dummy = _make_dummy_response_without_tool()

    async def request(self, *args, **kwargs):
        return dummy

    with patch.object(InstrumentedModel, "request", new=request):
        mlflow.autolog(log_traces=True)

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
    sequence, final_resp, usage_final = _make_dummy_response_with_tool()

    async def request(self, *args, **kwargs):
        if sequence:
            return sequence.pop(0)
        return final_resp, usage_final

    with patch.object(InstrumentedModel, "request", new=request):
        mlflow.autolog(log_traces=True)

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
    assert span3.name == "Tool.run"
    assert span3.span_type == SpanType.TOOL
    assert span3.parent_id == spans[1].span_id

    span4 = spans[4]
    assert span4.name == "InstrumentedModel.request_2"
    assert span4.span_type == SpanType.LLM
    assert span4.parent_id == spans[1].span_id


@pytest.mark.asyncio
async def test_agent_run_enable_disable_fluent_autolog_with_tool(agent_with_tool):
    sequence, final_resp, usage_final = _make_dummy_response_with_tool()

    async def request(self, *args, **kwargs):
        if sequence:
            return sequence.pop(0)
        return final_resp, usage_final

    with patch.object(InstrumentedModel, "request", new=request):
        mlflow.autolog(log_traces=True)

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
    assert span2.name == "Tool.run"
    assert span2.span_type == SpanType.TOOL
    assert span2.parent_id == spans[0].span_id

    span3 = spans[3]
    assert span3.name == "InstrumentedModel.request_2"
    assert span3.span_type == SpanType.LLM
    assert span3.parent_id == spans[0].span_id
