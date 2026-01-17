import sys
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
        command=sys.executable,
        args=[server.__file__],
        env={
            "MLFLOW_TRACKING_URI": mlflow.get_tracking_uri(),
            "MLFLOW_MCP_TOOLS": "all",  # Test all tools
        },
    )
    async with Client(transport) as client:
        yield client


@pytest.mark.asyncio
async def test_list_tools(client: Client):
    tools = await client.list_tools()
    assert sorted(t.name for t in tools) == [
        "build-docker_models",
        "create-endpoint_deployments",
        "create_deployments",
        "create_experiments",
        "create_runs",
        "csv_experiments",
        "delete-assessment_traces",
        "delete-endpoint_deployments",
        "delete-tag_traces",
        "delete_deployments",
        "delete_experiments",
        "delete_runs",
        "delete_traces",
        "describe_runs",
        "download_artifacts",
        "evaluate_traces",
        "explain_deployments",
        "generate-dockerfile_models",
        "get-assessment_traces",
        "get-endpoint_deployments",
        "get_deployments",
        "get_experiments",
        "get_traces",
        "help_deployments",
        "link-traces_runs",
        "list-endpoints_deployments",
        "list_artifacts",
        "list_deployments",
        "list_runs",
        "list_scorers",
        "log-artifact_artifacts",
        "log-artifacts_artifacts",
        "log-expectation_traces",
        "log-feedback_traces",
        "predict_deployments",
        "predict_models",
        "prepare-env_models",
        "register-llm-judge_scorers",
        "rename_experiments",
        "restore_experiments",
        "restore_runs",
        "run-local_deployments",
        "search_experiments",
        "search_traces",
        "serve_models",
        "set-tag_traces",
        "update-assessment_traces",
        "update-endpoint_deployments",
        "update-pip-requirements_models",
        "update_deployments",
    ]


@pytest.mark.asyncio
async def test_call_tool(client: Client):
    with mlflow.start_span() as span:
        pass

    result = await client.call_tool(
        "get_traces",
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
        "get_traces",
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
