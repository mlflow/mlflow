import importlib.metadata

import pytest
from packaging.version import Version

if Version(importlib.metadata.version("pydantic_ai")).major < 2:
    pytest.skip("Pydantic AI 2.x tracing tests", allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

import mlflow
from mlflow.entities import SpanType

from tests.tracing.helper import get_traces


def _span_names():
    traces = get_traces()
    assert len(traces) == 1
    return [span.name for span in traces[0].data.spans]


def test_autolog_enables_agent_instrumentation():
    mlflow.pydantic_ai.autolog()

    agent = Agent(TestModel())

    assert agent.instrument is True


def test_run_sync_preserves_run_nesting():
    mlflow.pydantic_ai.autolog()
    agent = Agent(TestModel(custom_output_text="hello"))

    assert agent.run_sync("hi").output == "hello"

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert [span.name for span in spans] == [
        "Agent.run_sync",
        "Agent.run",
        "TestModel.request",
    ]
    assert [span.span_type for span in spans] == [
        SpanType.AGENT,
        SpanType.AGENT,
        SpanType.LLM,
    ]
    assert spans[1].parent_id == spans[0].span_id
    assert spans[2].parent_id == spans[1].span_id


@pytest.mark.asyncio
async def test_run_is_traced():
    mlflow.pydantic_ai.autolog()
    agent = Agent(TestModel(custom_output_text="hello"))

    assert (await agent.run("hi")).output == "hello"
    assert _span_names() == ["Agent.run", "TestModel.request"]


@pytest.mark.asyncio
async def test_run_stream_is_traced():
    mlflow.pydantic_ai.autolog()
    agent = Agent(TestModel(custom_output_text="hello"))

    async with agent.run_stream("hi") as result:
        assert await result.get_output() == "hello"
    assert _span_names() == ["Agent.run_stream", "TestModel.request"]


@pytest.mark.parametrize(
    "completion_method",
    ["stream_text", "stream_output", "stream_response", "get_output"],
)
@pytest.mark.parametrize("use_context_manager", [True, False])
def test_run_stream_sync_lifecycle(completion_method, use_context_manager):
    mlflow.pydantic_ai.autolog()
    agent = Agent(TestModel(custom_output_text="hello"))

    result = agent.run_stream_sync("hi")

    def consume():
        value = getattr(result, completion_method)()
        return value if completion_method == "get_output" else list(value)

    if use_context_manager:
        with result:
            output = consume()
    else:
        output = consume()

    assert output
    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert [span.name for span in spans] == [
        "Agent.run_stream_sync",
        "Agent.run_stream",
        "TestModel.request",
    ]
    assert spans[1].parent_id == spans[0].span_id
    assert spans[2].parent_id == spans[1].span_id
