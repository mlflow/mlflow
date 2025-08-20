from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastmcp import Client

from mlflow.mcp import server


@pytest_asyncio.fixture()
async def client() -> AsyncIterator[Client]:
    async with Client(Path(server.__file__)) as client:
        yield client


@pytest.mark.asyncio
async def test_list_tools(client: Client):
    tools = await client.list_tools()
    assert len(tools) > 0


@pytest.mark.asyncio
async def test_call_tool(client: Client):
    result = await client.call_tool("test", {"a": "foo", "b": 2})
    assert result.content[0].text == "Test command called with a='foo' and b=2 and c='a'\n"
