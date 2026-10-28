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

from mlflow.entities.mcp_access_endpoint import MCPAccessEndpoint
from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPTool
from mlflow.entities.mcp_server_version import MCPServerVersion
from mlflow.exceptions import MlflowException
from mlflow.genai.mcp_tool_discovery import (
    _CLIENT_TIMEOUT_SLACK_SECONDS,
    DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS,
    _require_fastmcp,
    _run_coro_sync,
    _select_access_endpoints,
    _tool_from_sdk,
    discover_mcp_tools,
    discover_tools_for_version,
    resolve_tools_for_create,
)
from mlflow.store.tracking.mcp_server_registry.abstract_mixin import NOT_SET


def _endpoint(endpoint_id: str, version: str, creation_timestamp: int | None = None):
    return MCPAccessEndpoint(
        id=endpoint_id,
        server_name="io.github.test/server",
        url=f"https://mcp.example.com/{endpoint_id}",
        resolved_version=MCPServerVersion(
            name="io.github.test/server", version=version, server_json={}
        ),
        creation_timestamp=creation_timestamp,
    )


def test_select_access_endpoints_uses_earliest_creation_timestamp_then_id():
    endpoints = [
        _endpoint("later", "1.0.0", creation_timestamp=2),
        _endpoint("b", "1.0.0", creation_timestamp=1),
        _endpoint("a", "1.0.0", creation_timestamp=1),
        _endpoint("unknown-a", "1.0.0"),
        _endpoint("unknown", "1.0.0"),
        _endpoint("other", "2.0.0", creation_timestamp=0),
        MCPAccessEndpoint(id="unresolved", server_name="io.github.test/server", url="https://x"),
    ]

    assert [endpoint.id for endpoint in _select_access_endpoints(endpoints, "1.0.0")] == [
        "a",
        "b",
        "later",
        "unknown",
        "unknown-a",
    ]


@pytest.mark.parametrize("endpoints", [[], [_endpoint("other", "2.0.0")]])
def test_select_access_endpoints_returns_empty_when_no_matches(endpoints):
    assert _select_access_endpoints(endpoints, "1.0.0") == []


def test_discover_tools_for_version_prefers_matching_access_endpoint_and_headers():
    endpoint = _endpoint("endpoint", "1.0.0", creation_timestamp=1)
    with mock.patch(
        "mlflow.genai.mcp_tool_discovery.discover_mcp_tools", return_value=[MCPTool(name="tool")]
    ) as mock_discover:
        tools = discover_tools_for_version(
            {"remotes": [{"url": "https://mcp.example.com/remote"}]},
            "1.0.0",
            [endpoint],
            headers={"Authorization": "Bearer test"},
        )

    assert tools == [MCPTool(name="tool")]
    mock_discover.assert_called_once_with(
        url=endpoint.url,
        transport_type=MCPRemoteTransportType.STREAMABLE_HTTP,
        headers={"Authorization": "Bearer test"},
        timeout=DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS,
    )


@pytest.mark.parametrize("error_type", [MlflowException, RuntimeError])
@pytest.mark.parametrize("expected_tools", [[], [MCPTool(name="tool")]])
@pytest.mark.parametrize("failure_count", [0, 1, 2])
def test_discover_tools_for_version_tries_endpoints_in_order(
    error_type, expected_tools, failure_count
):
    endpoints = [
        _endpoint("first", "1.0.0", creation_timestamp=1),
        _endpoint("second", "1.0.0", creation_timestamp=2),
        _endpoint("third", "1.0.0", creation_timestamp=3),
    ]
    endpoints[1].transport_type = MCPRemoteTransportType.SSE
    headers = {"Authorization": "Bearer test"}
    with mock.patch(
        "mlflow.genai.mcp_tool_discovery.discover_mcp_tools",
        side_effect=[error_type("unavailable") for _ in range(failure_count)] + [expected_tools],
    ) as mock_discover:
        tools = discover_tools_for_version(
            {},
            "1.0.0",
            [_endpoint("other", "2.0.0", creation_timestamp=0), *endpoints[::-1]],
            headers=headers,
            timeout=2.0,
        )

    assert tools == expected_tools
    assert mock_discover.call_args_list == [
        mock.call(
            url=endpoint.url,
            transport_type=endpoint.transport_type,
            headers=headers,
            timeout=2.0,
        )
        for endpoint in endpoints[: failure_count + 1]
    ]


@pytest.mark.parametrize("error_type", [MlflowException, RuntimeError])
def test_discover_tools_for_version_raises_when_all_endpoints_fail(error_type):
    endpoints = [_endpoint("first", "1.0.0"), _endpoint("last", "1.0.0")]
    errors = [error_type("first unavailable"), error_type("last unavailable")]
    with (
        mock.patch(
            "mlflow.genai.mcp_tool_discovery.discover_mcp_tools",
            side_effect=errors,
        ) as mock_discover,
        pytest.raises(
            MlflowException,
            match="access endpoint 'last'.*https://mcp.example.com/last.*last unavailable",
        ) as exc_info,
    ):
        discover_tools_for_version(
            {"remotes": [{"url": "https://mcp.example.com/remote"}]}, "1.0.0", endpoints
        )

    assert mock_discover.call_args_list == [
        mock.call(
            url=endpoint.url,
            transport_type=endpoint.transport_type,
            headers=None,
            timeout=DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS,
        )
        for endpoint in endpoints
    ]
    assert exc_info.value.__cause__ is errors[-1]


def test_discover_tools_for_version_rejects_remote_fallback_when_endpoints_do_not_match():
    with (
        mock.patch("mlflow.genai.mcp_tool_discovery.discover_mcp_tools") as mock_discover,
        pytest.raises(MlflowException, match="server_json.remotes will not be used"),
    ):
        discover_tools_for_version(
            {"remotes": [{"url": "https://mcp.example.com/remote"}]},
            "1.0.0",
            [_endpoint("other", "2.0.0")],
        )
    mock_discover.assert_not_called()


def test_discover_tools_for_version_wraps_runtime_error_from_endpoint():
    endpoint = _endpoint("endpoint", "1.0.0")
    with (
        mock.patch(
            "mlflow.genai.mcp_tool_discovery.discover_mcp_tools",
            side_effect=RuntimeError("unavailable"),
        ) as mock_discover,
        pytest.raises(
            MlflowException,
            match="access endpoint 'endpoint'.*https://mcp.example.com/endpoint",
        ),
    ):
        discover_tools_for_version({}, "1.0.0", [endpoint])
    mock_discover.assert_called_once()


def test_discover_tools_for_version_enriches_mlflow_exception_from_endpoint():
    # When discover_mcp_tools raises a MlflowException that already embeds the
    # endpoint URL (e.g. a timeout message), the outer wrapper must not add the
    # URL again. The URL should appear exactly once in the final error message.
    endpoint = _endpoint("endpoint", "1.0.0")
    inner_msg = f"Timed out discovering MCP tools from {endpoint.url!r} after 10s"
    with (
        mock.patch(
            "mlflow.genai.mcp_tool_discovery.discover_mcp_tools",
            side_effect=MlflowException(inner_msg),
        ) as mock_discover,
        pytest.raises(MlflowException, match="access endpoint 'endpoint'") as exc_info,
    ):
        discover_tools_for_version({}, "1.0.0", [endpoint])
    mock_discover.assert_called_once()
    msg = str(exc_info.value)
    assert "access endpoint 'endpoint'" in msg
    assert inner_msg in msg
    assert msg.count(endpoint.url) == 1


def test_discover_tools_for_version_falls_back_to_remote_when_no_endpoints():
    with mock.patch(
        "mlflow.genai.mcp_tool_discovery.discover_mcp_tools",
        return_value=[MCPTool(name="tool")],
    ) as mock_discover:
        tools = discover_tools_for_version(
            {"remotes": [{"url": "https://mcp.example.com/remote"}]},
            "1.0.0",
            access_endpoints=None,
        )
    assert tools == [MCPTool(name="tool")]
    mock_discover.assert_called_once_with(
        url="https://mcp.example.com/remote",
        transport_type=MCPRemoteTransportType.STREAMABLE_HTTP,
        headers=None,
        timeout=DEFAULT_MCP_TOOL_DISCOVER_TIMEOUT_SECONDS,
    )


def test_discover_tools_for_version_raises_when_no_endpoints_and_no_remote():
    with pytest.raises(MlflowException, match="No usable MCP remote found"):
        discover_tools_for_version({"remotes": []}, "1.0.0", access_endpoints=None)


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
        "mlflow.genai.mcp_tool_discovery.discover_mcp_tools",
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
    with mock.patch("mlflow.genai.mcp_tool_discovery.discover_mcp_tools") as mock_discover:
        assert resolve_tools_for_create(sj, tools=None) is None
        assert resolve_tools_for_create(sj, tools=[]) == []
        assert resolve_tools_for_create(sj, tools=[MCPTool(name="x")])[0].name == "x"
    mock_discover.assert_not_called()


def test_resolve_tools_for_create_not_set_without_remote_returns_none():
    with mock.patch("mlflow.genai.mcp_tool_discovery.discover_mcp_tools") as mock_discover:
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
        "mlflow.genai.mcp_tool_discovery.discover_mcp_tools",
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
        "mlflow.genai.mcp_tool_discovery.discover_mcp_tools",
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
        "mlflow.genai.mcp_tool_discovery.discover_mcp_tools",
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
    with mock.patch("mlflow.genai.mcp_tool_discovery.discover_mcp_tools") as mock_discover:
        assert resolve_tools_for_create(sj, tools=NOT_SET) is None
    mock_discover.assert_not_called()


def test_resolve_tools_for_create_soft_fails_on_non_list_remotes():
    sj = {"name": "io.github.test/r", "version": "1.0.0", "remotes": "not-a-list"}
    with mock.patch("mlflow.genai.mcp_tool_discovery.discover_mcp_tools") as mock_discover:
        assert resolve_tools_for_create(sj, tools=NOT_SET) is None
    mock_discover.assert_not_called()


def test_resolve_tools_for_create_skips_discovery_when_env_disabled(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_MCP_TOOL_DISCOVERY", "false")
    sj = {
        "name": "io.github.test/r",
        "version": "1.0.0",
        "remotes": [{"type": "streamable-http", "url": "https://mcp.example.com/r"}],
    }
    with mock.patch("mlflow.genai.mcp_tool_discovery.discover_mcp_tools") as mock_discover:
        assert resolve_tools_for_create(sj, tools=NOT_SET) is None
        # Explicit values still win when discovery is disabled.
        assert resolve_tools_for_create(sj, tools=[MCPTool(name="manual")])[0].name == "manual"
    mock_discover.assert_not_called()


@pytest.fixture
def live_fastmcp_server_url():
    # Optional e2e scrape (real fastmcp Client + transport + Tool shape); skips
    # when mlflow[mcp] / fastmcp is not installed.
    pytest.importorskip("fastmcp")
    import uvicorn
    from fastmcp import FastMCP

    mcp = FastMCP(name="mlflow-discovery-it")

    @mcp.tool
    def echo(message: str) -> str:
        """Echo a message (integration fixture)."""
        return message

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                mcp.http_app(path="/mcp"),
                host="127.0.0.1",
                port=port,
                lifespan="on",
                ws="none",
                timeout_graceful_shutdown=2,
            )
        )
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [sock]},
            name="fastmcp-discovery-it",
            daemon=True,
        )
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started:
                if not thread.is_alive() or time.monotonic() >= deadline:
                    pytest.fail("FastMCP test server did not become ready")
                time.sleep(0.1)
            yield f"http://127.0.0.1:{port}/mcp"
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()


@pytest.mark.parametrize("fail_first", [False, True])
@pytest.mark.parametrize("running_loop", [False, True])
def test_discover_mcp_tools_against_live_fastmcp_server(
    live_fastmcp_server_url, fail_first, running_loop
):
    endpoint = _endpoint("live", "1.0.0", creation_timestamp=2)
    endpoint.url = live_fastmcp_server_url
    endpoints = [endpoint]
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        if fail_first:
            failed_endpoint = _endpoint("unavailable", "1.0.0", creation_timestamp=1)
            failed_endpoint.url = f"http://127.0.0.1:{unavailable.getsockname()[1]}/mcp"
            endpoints.append(failed_endpoint)

        def discover():
            return discover_tools_for_version({}, "1.0.0", endpoints, timeout=10)

        async def discover_from_running_loop():
            return discover()

        tools = asyncio.run(discover_from_running_loop()) if running_loop else discover()
    assert len(tools) == 1
    assert tools[0].name == "echo"
    assert tools[0].description
    assert tools[0].input_schema is not None
