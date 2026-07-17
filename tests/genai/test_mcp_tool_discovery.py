from __future__ import annotations

import asyncio
import socket
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPTool
from mlflow.exceptions import MlflowException
from mlflow.genai.mcp_tool_discovery import (
    _CLIENT_TIMEOUT_SLACK_SECONDS,
    DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS,
    _require_fastmcp,
    _run_coro_sync,
    _tool_from_sdk,
    discover_mcp_tools,
    resolve_tools_for_create,
)
from mlflow.store.tracking.mcp_server_registry.abstract_mixin import NOT_SET


def test_tool_from_sdk_model_dump():
    tool = SimpleNamespace(
        model_dump=lambda by_alias=True, exclude_none=True: {
            "name": "search",
            "description": "Search",
            "inputSchema": {"type": "object"},
            "meta": {"ignored": True},
        }
    )
    mapped = _tool_from_sdk(tool)
    assert mapped == MCPTool(
        name="search",
        description="Search",
        input_schema={"type": "object"},
    )


def test_tool_from_sdk_plain_object():
    tool = SimpleNamespace(
        name="echo",
        title="Echo",
        description="Echo input",
        inputSchema={"type": "object", "properties": {"text": {"type": "string"}}},
        outputSchema=None,
        annotations=None,
        icons=None,
        execution=None,
    )
    mapped = _tool_from_sdk(tool)
    assert mapped.name == "echo"
    assert mapped.title == "Echo"
    assert mapped.input_schema == {"type": "object", "properties": {"text": {"type": "string"}}}


def test_require_fastmcp_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastmcp", None)
    with pytest.raises(MlflowException, match="mlflow\\[mcp\\]"):
        _require_fastmcp()


def test_discover_mcp_tools_uses_streamable_http_transport():
    fake_tools = [
        SimpleNamespace(
            model_dump=lambda by_alias=True, exclude_none=True: {
                "name": "ping",
                "description": "Ping",
            }
        )
    ]
    seen: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, transport, timeout=None, init_timeout=None, **kwargs):
            self.transport = transport
            seen["timeout"] = timeout
            seen["init_timeout"] = init_timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def list_tools(self):
            return fake_tools

    class FakeTransport:
        def __init__(self, url, headers=None):
            self.url = url
            self.headers = headers

    with (
        mock.patch(
            "mlflow.genai.mcp_tool_discovery._require_fastmcp",
            return_value=(FakeClient, FakeTransport, FakeTransport),
        ),
    ):
        tools = discover_mcp_tools(
            url="https://mcp.example.com/server",
            transport_type=MCPRemoteTransportType.STREAMABLE_HTTP,
            headers={"Authorization": "Bearer x"},
        )

    assert len(tools) == 1
    assert tools[0].name == "ping"
    expected_client_timeout = (
        DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS + _CLIENT_TIMEOUT_SLACK_SECONDS
    )
    assert seen["timeout"] == expected_client_timeout
    assert seen["init_timeout"] == expected_client_timeout


def test_discover_mcp_tools_wraps_errors():
    class BoomClient:
        def __init__(self, transport, timeout=None, init_timeout=None, **kwargs):
            pass

        async def __aenter__(self):
            raise RuntimeError("connection refused")

        async def __aexit__(self, *args):
            return None

    class FakeTransport:
        def __init__(self, url, headers=None):
            pass

    with mock.patch(
        "mlflow.genai.mcp_tool_discovery._require_fastmcp",
        return_value=(BoomClient, FakeTransport, FakeTransport),
    ):
        with pytest.raises(MlflowException, match="Failed to discover MCP tools"):
            discover_mcp_tools(url="https://mcp.example.com/down")


def test_discover_mcp_tools_times_out():
    class FakeClient:
        def __init__(self, transport, timeout=None, init_timeout=None, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def list_tools(self):
            return []

    class FakeTransport:
        def __init__(self, url, headers=None):
            pass

    async def wait_for_timeout(coro, timeout=None):
        coro.close()
        raise asyncio.TimeoutError()

    with (
        mock.patch(
            "mlflow.genai.mcp_tool_discovery._require_fastmcp",
            return_value=(FakeClient, FakeTransport, FakeTransport),
        ),
        mock.patch(
            "mlflow.genai.mcp_tool_discovery.asyncio.wait_for",
            side_effect=wait_for_timeout,
        ),
    ):
        with pytest.raises(MlflowException, match="Timed out discovering MCP tools.*after 10"):
            discover_mcp_tools(url="https://mcp.example.com/slow")


def test_discover_mcp_tools_rejects_non_positive_timeout():
    with pytest.raises(MlflowException, match="timeout must be positive"):
        discover_mcp_tools(url="https://mcp.example.com/x", timeout=0)


def test_discover_mcp_tools_works_when_event_loop_already_running():
    fake_tools = [
        SimpleNamespace(
            model_dump=lambda by_alias=True, exclude_none=True: {"name": "ping"},
        )
    ]

    class FakeClient:
        def __init__(self, transport, timeout=None, init_timeout=None, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def list_tools(self):
            return fake_tools

    class FakeTransport:
        def __init__(self, url, headers=None):
            pass

    async def _call_from_running_loop():
        with mock.patch(
            "mlflow.genai.mcp_tool_discovery._require_fastmcp",
            return_value=(FakeClient, FakeTransport, FakeTransport),
        ):
            return discover_mcp_tools(url="https://mcp.example.com/server")

    tools = asyncio.run(_call_from_running_loop())
    assert tools[0].name == "ping"


def test_run_coro_sync_uses_worker_thread_when_loop_running():
    async def _add(a, b):
        return a + b

    async def _from_running_loop():
        return _run_coro_sync(_add(2, 3))

    assert asyncio.run(_from_running_loop()) == 5


def test_run_coro_sync_enforces_outer_timeout_when_loop_already_running():
    async def _hang():
        await asyncio.sleep(1.0)

    async def _from_running_loop():
        with pytest.raises(MlflowException, match="Timed out discovering MCP tools after 0.2"):
            _run_coro_sync(_hang(), timeout=0.2)

    t0 = time.monotonic()
    asyncio.run(_from_running_loop())
    assert time.monotonic() - t0 < 5.0


def test_resolve_tools_for_create_not_set_discovers_from_remote():
    sj = {
        "name": "io.github.test/r",
        "version": "1.0.0",
        "remotes": [{"type": "streamable-http", "url": "https://mcp.example.com/r"}],
    }
    with mock.patch(
        "mlflow.genai.mcp_tool_discovery._discover_mcp_tools",
        return_value=[MCPTool(name="t")],
    ) as mock_discover:
        tools = resolve_tools_for_create(sj, tools=NOT_SET, headers={"Authorization": "x"})
    mock_discover.assert_called_once_with(
        url="https://mcp.example.com/r",
        transport_type=MCPRemoteTransportType.STREAMABLE_HTTP,
        headers={"Authorization": "x"},
        timeout=DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS,
    )
    assert tools[0].name == "t"


def test_resolve_tools_for_create_explicit_none_and_list_skip_discovery():
    sj = {
        "name": "io.github.test/r",
        "version": "1.0.0",
        "remotes": [{"type": "streamable-http", "url": "https://mcp.example.com/r"}],
    }
    with mock.patch("mlflow.genai.mcp_tool_discovery._discover_mcp_tools") as mock_discover:
        assert resolve_tools_for_create(sj, tools=None) is None
        assert resolve_tools_for_create(sj, tools=[]) == []
        assert resolve_tools_for_create(sj, tools=[MCPTool(name="x")])[0].name == "x"
    mock_discover.assert_not_called()


def test_resolve_tools_for_create_not_set_without_remote_returns_none():
    with mock.patch("mlflow.genai.mcp_tool_discovery._discover_mcp_tools") as mock_discover:
        assert resolve_tools_for_create({"name": "n", "version": "1"}, tools=NOT_SET) is None
        assert (
            resolve_tools_for_create({"name": "n", "version": "1", "remotes": []}, tools=NOT_SET)
            is None
        )
    mock_discover.assert_not_called()


def test_resolve_tools_for_create_skips_remote_without_url_then_discovers_next():
    sj = {
        "name": "io.github.test/r",
        "version": "1.0.0",
        "remotes": [
            {"type": "streamable-http"},
            {"type": "sse", "url": "https://mcp.example.com/second"},
        ],
    }
    with mock.patch(
        "mlflow.genai.mcp_tool_discovery._discover_mcp_tools",
        return_value=[MCPTool(name="from-second")],
    ) as mock_discover:
        tools = resolve_tools_for_create(sj, tools=NOT_SET)
    mock_discover.assert_called_once_with(
        url="https://mcp.example.com/second",
        transport_type=MCPRemoteTransportType.SSE,
        headers=None,
        timeout=DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS,
    )
    assert tools[0].name == "from-second"


def test_resolve_tools_for_create_soft_fails_on_discovery_error_without_failover():
    # First eligible URL is scraped; scrape failure does not try later remotes.
    sj = {
        "name": "io.github.test/r",
        "version": "1.0.0",
        "remotes": [
            {"type": "streamable-http", "url": "https://mcp.example.com/dead"},
            {"type": "streamable-http", "url": "https://mcp.example.com/live"},
        ],
    }
    with mock.patch(
        "mlflow.genai.mcp_tool_discovery._discover_mcp_tools",
        side_effect=MlflowException.invalid_parameter_value("Failed to discover MCP tools"),
    ) as mock_discover:
        assert resolve_tools_for_create(sj, tools=NOT_SET) is None
    mock_discover.assert_called_once_with(
        url="https://mcp.example.com/dead",
        transport_type=MCPRemoteTransportType.STREAMABLE_HTTP,
        headers=None,
        timeout=DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS,
    )


def test_resolve_tools_for_create_skips_unsupported_transport_then_discovers_next():
    sj = {
        "name": "io.github.test/r",
        "version": "1.0.0",
        "remotes": [
            {"type": "grpc-bidirectional", "url": "https://mcp.example.com/bad"},
            {"type": "streamable-http", "url": "https://mcp.example.com/good"},
        ],
    }
    with mock.patch(
        "mlflow.genai.mcp_tool_discovery._discover_mcp_tools",
        return_value=[MCPTool(name="from-good")],
    ) as mock_discover:
        tools = resolve_tools_for_create(sj, tools=NOT_SET)
    mock_discover.assert_called_once_with(
        url="https://mcp.example.com/good",
        transport_type=MCPRemoteTransportType.STREAMABLE_HTTP,
        headers=None,
        timeout=DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS,
    )
    assert tools[0].name == "from-good"


def test_resolve_tools_for_create_soft_fails_when_only_unsupported_transport():
    sj = {
        "name": "io.github.test/r",
        "version": "1.0.0",
        "remotes": [{"type": "grpc-bidirectional", "url": "https://mcp.example.com/bad"}],
    }
    with mock.patch("mlflow.genai.mcp_tool_discovery._discover_mcp_tools") as mock_discover:
        assert resolve_tools_for_create(sj, tools=NOT_SET) is None
    mock_discover.assert_not_called()


def test_resolve_tools_for_create_soft_fails_on_non_list_remotes():
    sj = {"name": "io.github.test/r", "version": "1.0.0", "remotes": "not-a-list"}
    with mock.patch("mlflow.genai.mcp_tool_discovery._discover_mcp_tools") as mock_discover:
        assert resolve_tools_for_create(sj, tools=NOT_SET) is None
    mock_discover.assert_not_called()


def test_resolve_tools_for_create_skips_discovery_when_env_disabled(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_MCP_TOOL_DISCOVERY", "false")
    sj = {
        "name": "io.github.test/r",
        "version": "1.0.0",
        "remotes": [{"type": "streamable-http", "url": "https://mcp.example.com/r"}],
    }
    with mock.patch("mlflow.genai.mcp_tool_discovery._discover_mcp_tools") as mock_discover:
        assert resolve_tools_for_create(sj, tools=NOT_SET) is None
        # Explicit values still win when discovery is disabled.
        assert resolve_tools_for_create(sj, tools=[MCPTool(name="manual")])[0].name == "manual"
    mock_discover.assert_not_called()


def test_discover_mcp_tools_against_live_fastmcp_server():
    # Optional e2e scrape (real fastmcp Client + transport + Tool shape); skips
    # when mlflow[mcp] / fastmcp is not installed.
    pytest.importorskip("fastmcp")
    from fastmcp import FastMCP

    mcp = FastMCP(name="mlflow-discovery-it")

    @mcp.tool
    def echo(message: str) -> str:
        """Echo a message (integration fixture)."""
        return message

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    def _serve() -> None:
        try:
            mcp.run(
                transport="streamable-http",
                host="127.0.0.1",
                port=port,
                path="/mcp",
            )
        except TypeError:
            mcp.run(transport="http", host="127.0.0.1", port=port, path="/mcp")

    threading.Thread(target=_serve, name="fastmcp-discovery-it", daemon=True).start()
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    else:
        pytest.fail("FastMCP test server did not become ready")

    tools = discover_mcp_tools(
        url=f"http://127.0.0.1:{port}/mcp",
        transport_type=MCPRemoteTransportType.STREAMABLE_HTTP,
        timeout=10,
    )
    assert len(tools) == 1
    assert tools[0].name == "echo"
    assert tools[0].description
    assert tools[0].input_schema is not None
