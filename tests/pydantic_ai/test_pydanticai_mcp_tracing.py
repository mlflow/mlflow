from unittest.mock import patch

import pytest
from pydantic_ai.mcp import MCPServerStdio

import mlflow
from mlflow.entities.trace import SpanType

from tests.tracing.helper import get_traces


@pytest.mark.asyncio
async def test_mcp_server_list_tools_autolog():
    tools_list = [
        {"name": "tool1", "description": "Tool 1 description"},
        {"name": "tool2", "description": "Tool 2 description"},
    ]

    async def list_tools(self, *args, **kwargs):
        return tools_list

    with patch("pydantic_ai.mcp.MCPServer.list_tools", new=list_tools):
        mlflow.pydantic_ai.autolog(log_traces=True)

        server = MCPServerStdio(
            "deno",
            args=[
                "run",
                "-N",
                "-R=node_modules",
                "-W=node_modules",
                "--node-modules-dir=auto",
                "jsr:@pydantic/mcp-run-python",
                "stdio",
            ],
        )

        result = await server.list_tools()
        assert result == tools_list

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "MCPServerStdio.list_tools"
    assert span.span_type == SpanType.TOOL

    outputs = span.outputs
    assert len(outputs) == 2
    assert outputs == tools_list

    with patch("pydantic_ai.mcp.MCPServer.list_tools", new=list_tools):
        mlflow.pydantic_ai.autolog(disable=True)
        await server.list_tools()

    assert len(get_traces()) == 1


@pytest.mark.asyncio
async def test_mcp_server_call_tool_autolog():
    tool_name = "calculator"
    tool_args = {"operation": "add", "a": 5, "b": 7}
    tool_result = {"result": 12}

    async def call_tool(self, name, args, *remaining_args, **kwargs):
        assert name == tool_name
        assert args == tool_args
        return tool_result

    with patch("pydantic_ai.mcp.MCPServer.call_tool", new=call_tool):
        mlflow.pydantic_ai.autolog(log_traces=True)

        server = MCPServerStdio(
            "deno",
            args=[
                "run",
                "-N",
                "-R=node_modules",
                "-W=node_modules",
                "--node-modules-dir=auto",
                "jsr:@pydantic/mcp-run-python",
                "stdio",
            ],
        )

        result = await server.call_tool(tool_name, tool_args)

        assert result == tool_result

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert len(spans) == 1

    call_tool_span = spans[0]
    assert call_tool_span is not None
    assert call_tool_span.name == "MCPServerStdio.call_tool"
    assert call_tool_span.span_type == SpanType.TOOL

    inputs = call_tool_span.inputs
    assert len(inputs) == 2
    assert inputs["name"] == tool_name
    assert inputs["args"] == tool_args

    outputs = call_tool_span.outputs
    assert len(outputs) == 1
    assert outputs == tool_result

    with patch("pydantic_ai.mcp.MCPServer.call_tool", new=call_tool):
        mlflow.pydantic_ai.autolog(disable=True)
        await server.call_tool(tool_name, tool_args)

    assert len(get_traces()) == 1
