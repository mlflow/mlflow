from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

import mlflow
from mlflow.mcp import server


@pytest_asyncio.fixture()
async def client() -> AsyncIterator[Client]:
    transport = StdioTransport(
        command="python",
        args=[server.__file__],
        env={"MLFLOW_TRACKING_URI": mlflow.get_tracking_uri()},
    )
    async with Client(transport) as client:
        yield client


@pytest.mark.asyncio
async def test_list_tools(client: Client):
    tools = await client.list_tools()
    assert sorted(t.name for t in tools) == [
        "delete_assessment",
        "delete_trace_tag",
        "delete_traces",
        "get_assessment",
        "get_trace",
        "log_expectation",
        "log_feedback",
        "search_traces",
        "set_trace_tag",
        "update_assessment",
    ]


@pytest.mark.asyncio
async def test_call_tool(client: Client):
    with mlflow.start_span() as span:
        pass

    result = await client.call_tool(
        "get_trace",
        {"trace_id": span.trace_id},
        timeout=5,
    )
    assert span.trace_id in result.content[0].text

    experiment = mlflow.search_experiments(max_results=1)[0]
    result = await client.call_tool(
        "search_traces",
        {"experiment_id": experiment.experiment_id},
        timeout=5,
    )
    assert span.trace_id in result.content[0].text

    result = await client.call_tool(
        "delete_traces",
        {
            "experiment_id": experiment.experiment_id,
            "trace_ids": span.trace_id,
        },
        timeout=5,
    )
    result = await client.call_tool(
        "get_trace",
        {"trace_id": span.trace_id},
        timeout=5,
        raise_on_error=False,
    )
    assert result.is_error is True


@pytest.mark.asyncio
async def test_list_prompts(client: Client):
    prompts = await client.list_prompts()
    prompt_names = [p.name for p in prompts]

    # Should have at least the genai_analyze_experiment prompt
    assert "genai_analyze_experiment" in prompt_names

    # Find the analyze experiment prompt
    analyze_prompt = next(p for p in prompts if p.name == "genai_analyze_experiment")
    assert "experiment" in analyze_prompt.description.lower()
    assert "traces" in analyze_prompt.description.lower()


@pytest.mark.asyncio
async def test_get_prompt(client: Client):
    # Get the analyze experiment prompt
    result = await client.get_prompt("genai_analyze_experiment")

    # Should return messages
    assert len(result.messages) > 0

    # Content should contain the AI command instructions
    content = result.messages[0].content.text
    assert "Analyze Experiment" in content
    assert "Step 1: Setup and Configuration" in content
    assert "MLflow" in content
