import asyncio
from pathlib import Path

from fastmcp import Client

client = Client(Path(__file__).parent / "server.py")


async def call_tool(name: str):
    async with client:
        result = await client.call_tool("test", {"a": "foo", "b": 2})
        print(result)  # noqa: T201

        tools = await client.list_tools()
        print(tools)  # noqa: T201


asyncio.run(call_tool("Ford"))
