import sys

import pytest
from fastmcp import Client
from fastmcp.client.transports import ClientTransport, StdioTransport

import mlflow


@pytest.mark.parametrize(
    "transport",
    [
        StdioTransport(
            command=sys.executable,
            args=[
                "-m",
                "mlflow",
                "mcp",
                "run",
            ],
            env={"MLFLOW_TRACKING_URI": mlflow.get_tracking_uri()},
        ),
        StdioTransport(
            command="uv",
            args=[
                "run",
                "mlflow",
                "mcp",
                "run",
            ],
            env={"MLFLOW_TRACKING_URI": mlflow.get_tracking_uri()},
        ),
    ],
)
@pytest.mark.asyncio
async def test_cli(transport: ClientTransport):
    async with Client(transport) as client:
        tools = await client.list_tools()
        assert len(tools) > 0
