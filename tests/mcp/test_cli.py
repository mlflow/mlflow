import sys

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

import mlflow


@pytest.mark.asyncio
async def test_cli():
    transport = StdioTransport(
        command=sys.executable,
        args=[
            "-m",
            "mlflow",
            "mcp",
            "run",
        ],
        env={"MLFLOW_TRACKING_URI": mlflow.get_tracking_uri()},
    )
    async with Client(transport) as client:
        tools = await client.list_tools()
        assert len(tools) > 0
