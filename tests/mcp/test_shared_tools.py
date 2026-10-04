import contextlib
import io
import json
import os
import sys

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mlflow.mcp import server
from mlflow.mcp.tools import SHARED_TOOLS

from tests.mcp.helpers import SCENARIOS

_GENAI_CATEGORIES = ("traces", "scorers", "experiments", "runs")


class _StdoutGuard(io.TextIOBase):
    def __init__(self):
        self.writes: list[str] = []

    def write(self, s):
        self.writes.append(s)
        return len(s)


@pytest.fixture
def mcp():
    return server.create_mcp(categories=_GENAI_CATEGORIES)


@pytest.mark.asyncio
async def test_shared_tools_advertise_output_schemas_and_return_structured_content(mcp):
    shared = {tool.name for tool in SHARED_TOOLS}
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        assert shared <= tools.keys()
        for name in shared:
            assert tools[name].outputSchema["type"] == "object"
        # The stdio-only tool that still runs a CLI command returns text only.
        assert tools["evaluate_traces"].outputSchema is None

        for scenario in SCENARIOS:
            result = await client.call_tool(scenario.tool, scenario.setup("x"))
            assert json.loads(result.content[0].text) == result.structured_content


@pytest.mark.asyncio
async def test_no_shared_tool_touches_stdout(mcp, monkeypatch):
    # On stdio, anything written to stdout corrupts the protocol stream. The shared tools must
    # neither print nor swap ``sys.stdout`` to capture output.
    def forbidden(*args, **kwargs):
        raise AssertionError("shared tools must not redirect stdout or stderr")

    monkeypatch.setattr(contextlib, "redirect_stdout", forbidden)
    monkeypatch.setattr(contextlib, "redirect_stderr", forbidden)
    async with Client(mcp) as client:
        for scenario in SCENARIOS:
            arguments = scenario.setup("x")
            guard = _StdoutGuard()
            monkeypatch.setattr(sys, "stdout", guard)
            try:
                await client.call_tool(scenario.tool, arguments)
                assert sys.stdout is guard
            finally:
                monkeypatch.setattr(sys, "stdout", sys.__stdout__)
            assert guard.writes == [], scenario.tool


@pytest.mark.asyncio
async def test_search_traces_rejects_sql_warehouse_id(mcp, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", raising=False)
    async with Client(mcp) as client:
        tool = next(t for t in await client.list_tools() if t.name == "search_traces")
        assert "sql_warehouse_id" not in tool.inputSchema["properties"]
        with pytest.raises(ToolError, match="sql_warehouse_id"):
            await client.call_tool(
                "search_traces", {"experiment_id": "0", "sql_warehouse_id": "abc"}
            )
    assert "MLFLOW_TRACING_SQL_WAREHOUSE_ID" not in os.environ
