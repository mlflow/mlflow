import asyncio
import inspect
import json
import os
import pickle
import subprocess
import sys
import warnings
from contextlib import contextmanager, nullcontext, suppress
from copy import copy, deepcopy
from textwrap import dedent
from threading import BoundedSemaphore, Event
from unittest import mock

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from packaging.version import Version
from starlette.responses import PlainTextResponse

import mlflow
from mlflow.entities import MCPAccessEndpoint, MCPServerRef, SpanStatusCode, SpanType
from mlflow.entities.mcp_server import MCPRemoteTransportType
from mlflow.langchain import _mcp
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracking._tracking_service.utils import _get_store, _use_tracking_uri
from mlflow.utils.workspace_context import ServerWorkspaceContext, get_request_workspace
from mlflow.version import IS_TRACING_SDK_ONLY

from tests.helper_functions import get_safe_port

_SERVER_NAME = "io.acme/weather"
_URL = "https://weather.example.com/mcp"
_TRANSPORT = MCPRemoteTransportType.STREAMABLE_HTTP
pytestmark = pytest.mark.skipif(IS_TRACING_SDK_ONLY, reason="Requires the full MLflow test backend")


def _mock_endpoints(*endpoints):
    return mock.patch.object(
        _get_store(), "search_mcp_access_endpoints", return_value=PagedList(list(endpoints), None)
    )


def _mcp_attributes(span):
    return {key: value for key, value in span.attributes.items() if key.startswith("mlflow.mcp.")}


def _endpoint(**kwargs):
    return MCPAccessEndpoint(**{
        "id": "ae-weather",
        "server_name": _SERVER_NAME,
        "url": _URL,
        **kwargs,
    })


@pytest.mark.parametrize("remote_type", [_TRANSPORT, None])
def test_capture_adapter_url(remote_type):
    transport = mock.NonCallableMock(url=_URL, spec=["url", "_mlflow_track_mcp_session"])
    adapter = mock.NonCallableMock(client=mock.NonCallableMock(transport=transport))
    del adapter._mlflow_mcp_url
    original = mock.Mock(return_value=None)
    with mock.patch.object(_mcp, "_remote_transport_type", return_value=remote_type) as get_type:
        assert _mcp._patched_mcp_adapter_init(original, adapter, _URL) is None
    original.assert_called_once_with(adapter, _URL)
    get_type.assert_called_once_with(transport)
    if remote_type is not None:
        assert adapter._mlflow_mcp_url == _URL
        assert transport._mlflow_track_mcp_session is True
    else:
        assert not hasattr(adapter, "_mlflow_mcp_url")


@pytest.mark.parametrize("version", ["1.3.0", "1.4.1rc1", "1.5.1"])
def test_autolog_skips_unsupported_mcp_versions(monkeypatch, version):
    monkeypatch.setattr(_mcp, "get_installed_version", lambda package: Version(version))
    with mock.patch.object(_mcp, "safe_patch") as patch:
        mlflow.langchain.autolog()
    patch.assert_not_called()


def test_autolog_without_mcp(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain.mcp", None)
    monkeypatch.setattr(_mcp, "get_installed_version", lambda package: Version("1.4.0"))
    mlflow.langchain.autolog()

    @tool
    def forecast() -> str:
        """Get the forecast."""
        return "sunny"

    assert (
        forecast.invoke({}, config={"metadata": {_mcp._MCP_TOOL_METADATA_KEY: "caller value"}})
        == "sunny"
    )
    span = mlflow.get_trace(mlflow.get_last_active_trace_id()).data.spans[0]
    assert span.name == "forecast"
    assert span.span_type == SpanType.TOOL
    assert span.attributes["metadata"] == {_mcp._MCP_TOOL_METADATA_KEY: "caller value"}


def test_tool_metadata_marker_survives_copy_and_pickles_as_id():
    origin = _mcp._MCPToolOrigin(MCPServerRef(_SERVER_NAME), "sqlite://", "default")
    transport = mock.NonCallableMock(url=_URL)
    client = mock.NonCallableMock(transport=transport)
    marker = _mcp._MCPToolMetadata(origin, client, transport, _URL, tool_name="forecast")

    assert copy(marker) is marker
    assert deepcopy({"metadata": marker})["metadata"] is marker
    restored = pickle.loads(pickle.dumps(marker))
    assert type(restored) is str
    assert restored == marker


@pytest.mark.parametrize("second_page", ["match", "duplicate", "error", "no_match"])
@pytest.mark.asyncio
async def test_lookup_all_endpoint_pages(second_page):
    endpoint = _endpoint(workspace="weather")
    first_page = PagedList(
        [
            _endpoint(url=f"{_URL}/other"),
            _endpoint(transport_type=MCPRemoteTransportType.SSE),
            *([endpoint] if second_page in {"duplicate", "error"} else []),
        ],
        "next-page",
    )
    last_page = {
        "match": PagedList([endpoint], None),
        "duplicate": PagedList([endpoint], None),
        "error": RuntimeError("Registry unavailable"),
        "no_match": PagedList([], None),
    }[second_page]

    def search_page(*, page_token):
        assert get_request_workspace() == "weather"
        if page_token is None:
            return first_page
        if isinstance(last_page, Exception):
            raise last_page
        return last_page

    with (
        ServerWorkspaceContext("other"),
        mock.patch.object(
            _get_store(), "search_mcp_access_endpoints", side_effect=search_page
        ) as search,
    ):
        lookup = asyncio.to_thread(
            _mcp._find_mcp_tool_origin, mlflow.get_tracking_uri(), _URL, _TRANSPORT, "weather"
        )
        if second_page == "error":
            with pytest.raises(RuntimeError, match="Registry unavailable"):
                await lookup
        elif second_page == "match":
            origin = await lookup
            assert origin.server == MCPServerRef(_SERVER_NAME)
            assert origin.workspace == "weather"
        else:
            assert await lookup is None
        assert get_request_workspace() == "other"
    assert search.call_args_list == [mock.call(page_token=None), mock.call(page_token="next-page")]


@pytest.mark.parametrize(
    ("version", "same_server", "fail"),
    [(None, True, False), ("1.0.0", True, False), ("1.0.0", False, False), ("1.0.0", True, True)],
)
@pytest.mark.asyncio
async def test_tool_span_preserves_metadata_and_uses_tool_name(version, same_server, fail):
    session = object()

    @tool
    async def forecast() -> str:
        """Get the forecast."""
        async with _mcp._link_session_tool_call(session, _mcp._url_digest(_URL), "forecast"):
            if fail:
                raise ValueError("Forecast unavailable")
            return "sunny"

    tracking_uri = mlflow.get_tracking_uri() if same_server else "http://other.example.com/"
    origin = _mcp._MCPToolOrigin(
        MCPServerRef(_SERVER_NAME, version=version),
        tracking_uri,
        "default",
    )
    transport = mock.NonCallableMock(url=_URL)
    client = mock.NonCallableMock(transport=transport, session=session, is_connected=lambda: True)
    marker = _mcp._MCPToolMetadata(origin, client, transport, _URL, tool_name="forecast")
    forecast.metadata = {"tool_metadata": "preserved", _mcp._MCP_TOOL_METADATA_KEY: marker}

    mlflow.langchain.autolog()
    call = forecast.ainvoke(
        {}, config={"run_name": "custom forecast", "metadata": {"caller_metadata": "preserved"}}
    )
    with ServerWorkspaceContext("default"):
        if fail:
            with pytest.raises(ValueError, match="Forecast unavailable"):
                await call
        else:
            assert await call == "sunny"
    span = mlflow.get_trace(mlflow.get_last_active_trace_id()).data.spans[0]
    assert span.name == "custom forecast"
    assert span.status.status_code == (SpanStatusCode.ERROR if fail else SpanStatusCode.OK)
    assert span.attributes["metadata"] == {
        "tool_metadata": "preserved",
        "caller_metadata": "preserved",
    }
    assert forecast.metadata[_mcp._MCP_TOOL_METADATA_KEY] is marker
    expected = {
        SpanAttributeKey.MCP_SERVER_NAME: _SERVER_NAME,
        SpanAttributeKey.MCP_TOOL_NAME: "forecast",
    }
    if version is not None:
        expected[SpanAttributeKey.MCP_SERVER_VERSION] = version
    assert _mcp_attributes(span) == (expected if same_server else {})


@pytest.fixture
def mcp_adapter_class():
    if os.environ.get("MLFLOW_TEST_MCP_TRACKING_PYTHON"):
        from langchain.mcp import MCPAdapter

        return MCPAdapter
    adapter = pytest.importorskip("langchain.mcp", exc_type=ImportError).MCPAdapter
    if not inspect.iscoroutinefunction(adapter.list_tools):
        pytest.skip("Async MCPAdapter is unavailable")
    return adapter


@pytest.fixture
def serve_weather(mcp_adapter_class):
    from fastmcp import FastMCP
    from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

    from tests.tracking.integration_test_utils import ServerThread

    @contextmanager
    def serve(
        label="Sunny", *, transport="streamable-http", calls=None, wait_for_tool=None, token=None
    ):
        auth = StaticTokenVerifier({token: {"client_id": "weather-client"}}) if token else None
        server = FastMCP("Weather", auth=auth)

        @server.tool
        async def forecast(city: str) -> str:
            """Get the forecast."""
            if calls is not None:
                calls.append(city)
            if wait_for_tool is not None:
                await wait_for_tool()
            return f"{label} in {city}"

        @server.custom_route("/health", methods=["GET"])
        async def health(request):
            return PlainTextResponse("OK")

        path = "/sse" if transport == "sse" else "/mcp"
        with ServerThread(server.http_app(path=path, transport=transport), get_safe_port()) as uri:
            yield f"{uri}{path}"

    return serve


@pytest.fixture
def mcp_url(serve_weather):
    with serve_weather() as uri:
        yield uri


def test_autolog_installs_adapter_patches_with_strict_beta_warnings(mcp_adapter_class):
    subprocess.check_call([
        sys.executable,
        "-c",
        dedent("""
                import sys
                import warnings

                import mlflow.langchain
                from langchain_core._api import LangChainBetaWarning

                assert "langchain.mcp" not in sys.modules
                warnings.simplefilter("error", LangChainBetaWarning)
                mlflow.langchain.autolog()
                from langchain.mcp import MCPAdapter

                url = "https://weather.example.com/mcp"
                assert MCPAdapter(url)._mlflow_mcp_url == url
                """),
    ])


@pytest.mark.parametrize("exclusive", [True, False])
@pytest.mark.asyncio
async def test_concurrent_discovery_preserves_ordinary_tool_tracing(mcp_adapter_class, exclusive):
    discovery_started = asyncio.Event()
    release_discovery = asyncio.Event()

    async def list_tools(self, *, cache_mode):
        if cache_mode == "bypass":
            raise RuntimeError("MCP discovery failed")
        discovery_started.set()
        await release_discovery.wait()
        return []

    @tool
    def local_tool() -> str:
        """Return a local result."""
        return "ok"

    with mock.patch.object(mcp_adapter_class, "list_tools", list_tools):
        mlflow.langchain.autolog(exclusive=exclusive)
        adapter = mcp_adapter_class(_URL)
        pending = asyncio.create_task(adapter.list_tools(cache_mode="refresh"))
        try:
            await asyncio.wait_for(discovery_started.wait(), timeout=5)
            if not exclusive:
                with pytest.raises(RuntimeError, match="MCP discovery failed"):
                    await adapter.list_tools(cache_mode="bypass")
            with mlflow.start_run() if exclusive else nullcontext():
                assert local_tool.invoke({}) == "ok"
                trace_id = mlflow.get_last_active_trace_id()
                if exclusive:
                    assert trace_id is None
                else:
                    assert mlflow.get_trace(trace_id, flush=True).data.spans[0].name == "local_tool"
        finally:
            release_discovery.set()
            await asyncio.wait_for(pending, timeout=5)
            mlflow.langchain.autolog(disable=True)


@pytest.mark.asyncio
async def test_concurrent_discovery_preserves_warning_handling(mcp_adapter_class, mcp_url):
    original = mcp_adapter_class.list_tools
    lookup_started = asyncio.Event()
    release_lookup = asyncio.Event()
    second_discovery_started = asyncio.Event()
    release_second_discovery = asyncio.Event()

    async def list_tools(self, *, cache_mode):
        if cache_mode == "bypass":
            second_discovery_started.set()
            await release_second_discovery.wait()
            warnings.warn("MCP test warning from discovery", stacklevel=2)
        return await original(self, cache_mode=cache_mode)

    async def lookup(*args):
        lookup_started.set()
        await release_lookup.wait()

    with (
        mock.patch.object(mcp_adapter_class, "list_tools", list_tools),
        mock.patch.object(_mcp, "_lookup", side_effect=lookup) as search,
    ):
        mlflow.langchain.autolog(silent=True)
        try:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", UserWarning)
                async with mcp_adapter_class(mcp_url) as adapter:
                    tasks = [asyncio.create_task(adapter.list_tools(cache_mode="refresh"))]
                    try:
                        await asyncio.wait_for(lookup_started.wait(), timeout=5)
                        warnings.warn("MCP test warning during lookup", stacklevel=2)
                        tasks.append(asyncio.create_task(adapter.list_tools(cache_mode="bypass")))
                        await asyncio.wait_for(second_discovery_started.wait(), timeout=5)
                        release_lookup.set()
                        await asyncio.wait_for(tasks[0], timeout=5)
                        release_second_discovery.set()
                        await asyncio.wait_for(tasks[1], timeout=5)
                        warnings.warn("MCP test warning after discovery", stacklevel=2)
                    finally:
                        release_lookup.set()
                        release_second_discovery.set()
                        await asyncio.gather(*tasks, return_exceptions=True)
            assert [str(w.message) for w in recorded if str(w.message).startswith("MCP test")] == [
                "MCP test warning during lookup",
                "MCP test warning from discovery",
                "MCP test warning after discovery",
            ]
        finally:
            mlflow.langchain.autolog(disable=True)
    assert search.await_count == 2


@pytest.mark.asyncio
async def test_exclusive_autologging_links_tools_discovered_during_manual_run(
    mcp_adapter_class, mcp_url
):
    mlflow.langchain.autolog(exclusive=True, run_tracer_inline=True)
    endpoint = _endpoint(url=mcp_url, workspace="default")
    with _mock_endpoints(endpoint) as search:
        with mlflow.start_run():
            [forecast] = await mcp_adapter_class(mcp_url).list_tools()
            assert "Sunny in Boston" in str(await forecast.ainvoke({"city": "Boston"}))
            assert mlflow.get_last_active_trace_id() is None
    search.assert_called_once_with(page_token=None)

    assert "Sunny in Boston" in str(await forecast.ainvoke({"city": "Boston"}))
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id(), flush=True)
    [span] = trace.data.spans
    assert span.attributes[SpanAttributeKey.MCP_SERVER_NAME] == _SERVER_NAME
    assert span.attributes[SpanAttributeKey.MCP_TOOL_NAME] == "forecast"


@pytest.mark.parametrize("transport", [_TRANSPORT, MCPRemoteTransportType.SSE])
@pytest.mark.asyncio
async def test_tool_uses_actual_mcp_session(mcp_adapter_class, serve_weather, transport):
    mlflow.langchain.autolog()
    with (
        serve_weather(transport=transport.value) as discovered_url,
        serve_weather("Rainy", transport=transport.value) as other_url,
    ):
        adapter = mcp_adapter_class(discovered_url)
        endpoint = _endpoint(url=discovered_url, transport_type=transport)
        with _mock_endpoints(endpoint) as search:
            async with adapter:
                [forecast] = await adapter.list_tools()
                assert "Sunny in Boston" in str(await forecast.ainvoke({"city": "Boston"}))
                span = mlflow.get_trace(mlflow.get_last_active_trace_id()).data.spans[0]
                assert span.attributes[SpanAttributeKey.MCP_SERVER_NAME] == _SERVER_NAME

            adapter.client.transport.url = other_url
            async with adapter:
                adapter.client.transport.url = discovered_url
                [rediscovered] = await adapter.list_tools()
                assert _mcp._MCP_TOOL_METADATA_KEY not in (rediscovered.metadata or {})
                assert "Rainy in Boston" in str(await forecast.ainvoke({"city": "Boston"}))
                span = mlflow.get_trace(mlflow.get_last_active_trace_id()).data.spans[0]
                assert SpanAttributeKey.MCP_SERVER_NAME not in span.attributes

    search.assert_called_once_with(page_token=None)


@pytest.mark.parametrize("transport", [_TRANSPORT, MCPRemoteTransportType.SSE])
@pytest.mark.asyncio
async def test_deep_copied_adapter_connects_to_its_own_url(
    mcp_adapter_class, serve_weather, transport
):
    mlflow.langchain.autolog()
    with (
        serve_weather(transport=transport.value) as first_url,
        serve_weather("Rainy", transport=transport.value) as second_url,
    ):
        original = mcp_adapter_class(first_url)
        copied = deepcopy(original)
        copied.client.transport.url = second_url

        assert original.client.transport.url == first_url
        [forecast] = await copied.list_tools()
        assert "Rainy in Boston" in str(await forecast.ainvoke({"city": "Boston"}))


@pytest.mark.parametrize("target_type", ["transport", "prebuilt_client"])
@pytest.mark.parametrize("transport_type", [_TRANSPORT, MCPRemoteTransportType.SSE])
@pytest.mark.asyncio
async def test_authenticated_single_endpoint_links(
    mcp_adapter_class, serve_weather, target_type, transport_type
):
    from fastmcp import Client
    from fastmcp.client.transports import SSETransport, StreamableHttpTransport

    class MetadataCallback(BaseCallbackHandler):
        run_inline = True
        raise_error = True

        def on_tool_start(self, serialized, input_str, *, metadata, **kwargs):
            callback_metadata.append(json.dumps(metadata))

    mlflow.langchain.autolog()
    token = "weather-test-bearer-token"
    callback_metadata = []
    with serve_weather(transport=transport_type.value, token=token) as url:
        sse = transport_type == MCPRemoteTransportType.SSE
        transport_class = SSETransport if sse else StreamableHttpTransport
        target = (
            transport_class(url, auth=token)
            if target_type == "transport"
            else Client(url, auth=token)
        )
        adapter = mcp_adapter_class(target)
        endpoint = _endpoint(url=url, transport_type=transport_type)
        with _mock_endpoints(endpoint) as search:
            [forecast] = await adapter.list_tools()
        search.assert_called_once_with(page_token=None)

        with ServerWorkspaceContext("default"):
            result = await forecast.ainvoke(
                {"city": "Boston"}, config={"callbacks": [MetadataCallback()]}
            )
    assert "Sunny in Boston" in str(result)
    assert len(callback_metadata) == 1
    assert token not in callback_metadata[0]
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id(), flush=True)
    [span] = trace.data.spans
    assert span.attributes[SpanAttributeKey.MCP_SERVER_NAME] == _SERVER_NAME
    assert span.attributes[SpanAttributeKey.MCP_TOOL_NAME] == "forecast"
    assert token not in json.dumps(span.attributes)


@pytest.mark.asyncio
async def test_prebuilt_client_requires_a_session_with_known_endpoint(mcp_adapter_class, mcp_url):
    from fastmcp import Client

    async def elicit(message, response_type, params, context):
        raise AssertionError("The test server does not request input")

    mlflow.langchain.autolog()
    client = Client(mcp_url, elicitation_handler=elicit)
    with _mock_endpoints(_endpoint(url=mcp_url)) as search:
        async with client:
            adapter = mcp_adapter_class(client)
            assert adapter.client is client
            [forecast] = await adapter.list_tools()
            assert _mcp._MCP_TOOL_METADATA_KEY not in (forecast.metadata or {})
            search.assert_not_called()

        [forecast] = await adapter.list_tools()
        assert _mcp._MCP_TOOL_METADATA_KEY in forecast.metadata
        search.assert_called_once_with(page_token=None)


@pytest.mark.parametrize("case", ["registry_error", "config", "changed_url", "disabled"])
@pytest.mark.asyncio
async def test_adapter_discovery_without_link(mcp_adapter_class, mcp_url, case):
    target = mcp_url
    if case == "config":
        target = {"mcpServers": {"weather": {"url": mcp_url}}}

    mlflow.langchain.autolog(log_traces=case != "disabled")
    adapter = mcp_adapter_class(target)
    if case == "changed_url":
        adapter.client.transport.url = f"{mcp_url}?instance=second"
    with mock.patch.object(
        _get_store(),
        "search_mcp_access_endpoints",
        side_effect=[
            PagedList([_endpoint(url=mcp_url)], "next-page"),
            RuntimeError("Registry unavailable"),
        ],
    ) as search:
        [forecast] = await adapter.list_tools(cache_mode="bypass")
        assert _mcp._MCP_TOOL_METADATA_KEY not in (forecast.metadata or {})
        assert "Sunny in Boston" in str(await forecast.ainvoke({"city": "Boston"}))

    if case == "registry_error":
        assert search.call_args_list == [
            mock.call(page_token=None),
            mock.call(page_token="next-page"),
        ]
    else:
        search.assert_not_called()
    if case == "disabled":
        assert mlflow.get_last_active_trace_id() is None
    else:
        span = mlflow.get_trace(mlflow.get_last_active_trace_id()).data.spans[0]
        assert span.span_type == SpanType.TOOL
        assert SpanAttributeKey.MCP_SERVER_NAME not in span.attributes


@pytest.mark.asyncio
async def test_adapters_with_overlapping_tool_names(mcp_adapter_class, mcp_url):
    second_url = f"{mcp_url}?instance=second"
    second_server = "io.acme/other-weather"
    mlflow.langchain.autolog()
    with _mock_endpoints(
        _endpoint(url=mcp_url), _endpoint(url=second_url, server_name=second_server)
    ) as search:
        [first] = await mcp_adapter_class(mcp_url).list_tools()
        [second] = await mcp_adapter_class(second_url).list_tools()
    assert search.call_args_list == [mock.call(page_token=None), mock.call(page_token=None)]

    for forecast, server_name in [(first, _SERVER_NAME), (second, second_server)]:
        assert forecast.name == "forecast"
        assert "Sunny in Boston" in str(await forecast.ainvoke({"city": "Boston"}))
        span = mlflow.get_trace(mlflow.get_last_active_trace_id()).data.spans[0]
        assert span.attributes[SpanAttributeKey.MCP_SERVER_NAME] == server_name


@pytest.mark.parametrize("mutate_connection", [None, "before_invocation", "in_callback"])
@pytest.mark.asyncio
async def test_discovered_tool_with_json_callback(mcp_adapter_class, mcp_url, mutate_connection):
    class JsonCallback(BaseCallbackHandler):
        raise_error = True
        run_inline = True

        def on_tool_start(self, serialized, input_str, *, metadata, **kwargs):
            observed_metadata.append(json.loads(json.dumps(metadata)))
            if mutate_connection == "in_callback":
                adapter.client.transport.url = f"{mcp_url}?instance=second"

    observed_metadata = []
    mlflow.langchain.autolog(run_tracer_inline=True)
    adapter = mcp_adapter_class(mcp_url)
    with _mock_endpoints(_endpoint(url=mcp_url)) as search:
        [forecast] = await adapter.list_tools()
    search.assert_called_once_with(page_token=None)

    if mutate_connection == "before_invocation":
        adapter.client.transport.url = f"{mcp_url}?instance=second"
    with ServerWorkspaceContext("default"):
        result = await forecast.ainvoke(
            {"city": "Boston"},
            config={"metadata": {"person": "Alice"}, "callbacks": [JsonCallback()]},
        )
    assert "Sunny in Boston" in str(result)
    [metadata] = observed_metadata
    assert metadata["person"] == "Alice"
    assert isinstance(metadata[_mcp._MCP_TOOL_METADATA_KEY], str)
    span = mlflow.get_trace(mlflow.get_last_active_trace_id()).data.spans[0]
    assert span.attributes.get(SpanAttributeKey.MCP_SERVER_NAME) == (
        None if mutate_connection else _SERVER_NAME
    )
    assert _mcp._MCP_TOOL_METADATA_KEY not in span.attributes["metadata"]


@pytest.mark.asyncio
async def test_discovery_times_out_without_waiting_for_store(
    mcp_adapter_class, mcp_url, monkeypatch
):
    mlflow.langchain.autolog()
    monkeypatch.setattr(_mcp, "_LOOKUP_TIMEOUT_SECONDS", 0.1)
    release_search = Event()

    def slow_search(**kwargs):
        assert release_search.wait(timeout=5)
        return PagedList([_endpoint(url=mcp_url)], None)

    with mock.patch.object(
        _get_store(), "search_mcp_access_endpoints", side_effect=slow_search
    ) as search:
        try:
            [forecast] = await mcp_adapter_class(mcp_url).list_tools()
            assert _mcp._MCP_TOOL_METADATA_KEY not in (forecast.metadata or {})
        finally:
            release_search.set()
    search.assert_called_once_with(page_token=None)


def test_timed_out_lookup_does_not_delay_event_loop_shutdown(monkeypatch):
    monkeypatch.setattr(_mcp, "_LOOKUP_TIMEOUT_SECONDS", 0.25)
    lookup_started = Event()
    release_lookup = Event()
    lookup_finished = Event()

    def slow_lookup(tracking_uri):
        lookup_started.set()
        try:
            release_lookup.wait(timeout=5)
        finally:
            lookup_finished.set()

    try:
        assert asyncio.run(_mcp._lookup(slow_lookup, mlflow.get_tracking_uri())) is None
        assert lookup_started.is_set()
        assert not lookup_finished.is_set()
    finally:
        release_lookup.set()
        assert lookup_finished.wait(timeout=5)


@pytest.mark.asyncio
async def test_concurrent_lookups_wait_for_capacity(monkeypatch):
    first_lookup_started = asyncio.Event()
    waiting_for_capacity = asyncio.Event()
    release_first_lookup = Event()
    loop = asyncio.get_running_loop()
    capacity = BoundedSemaphore(1)

    def acquire(*, blocking):
        acquired = capacity.acquire(blocking=blocking)
        if not acquired:
            waiting_for_capacity.set()
        return acquired

    def lookup(uri, name):
        if name == "first":
            loop.call_soon_threadsafe(first_lookup_started.set)
            assert release_first_lookup.wait(timeout=5)
        return name

    with mock.patch.object(
        _mcp, "_lookup_slots", wraps=capacity, **{"acquire.side_effect": acquire}
    ) as slots:
        monkeypatch.setattr(_mcp, "_LOOKUP_TIMEOUT_SECONDS", 3)
        tracking_uri = mlflow.get_tracking_uri()
        tasks = [asyncio.create_task(_mcp._lookup(lookup, tracking_uri, "first"))]
        try:
            await asyncio.wait_for(first_lookup_started.wait(), timeout=2)
            tasks.append(asyncio.create_task(_mcp._lookup(lookup, tracking_uri, "second")))
            await asyncio.wait_for(waiting_for_capacity.wait(), timeout=2)
            release_first_lookup.set()
            assert await asyncio.gather(*tasks) == ["first", "second"]
        finally:
            release_first_lookup.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        slots.acquire.assert_called_with(blocking=False)


@pytest.mark.parametrize("run_tracer_inline", [True, False])
@pytest.mark.parametrize("lookup_finishes", [True, False])
@pytest.mark.asyncio
async def test_tool_waits_for_pending_workspace_lookup(
    mcp_adapter_class, mcp_url, run_tracer_inline, lookup_finishes, monkeypatch
):
    mlflow.langchain.autolog(run_tracer_inline=run_tracer_inline)
    with _mock_endpoints(_endpoint(url=mcp_url)) as search:
        [forecast] = await mcp_adapter_class(mcp_url).list_tools()
    search.assert_called_once_with(page_token=None)

    monkeypatch.setattr(_mcp, "_LOOKUP_TIMEOUT_SECONDS", 5 if lookup_finishes else 0.5)
    release_lookup = Event()
    loop = asyncio.get_running_loop()

    def get_workspace(*args):
        # Finish only after the tool has returned its result to the session wrapper.
        assert release_lookup.wait(timeout=5)
        return "default"

    async def release_after_tool():
        await asyncio.sleep(0.2)
        release_lookup.set()

    with mock.patch.object(_mcp, "_experiment_workspace", side_effect=get_workspace) as lookup:
        releaser = asyncio.create_task(release_after_tool()) if lookup_finishes else None
        started = loop.time()
        try:
            result = await asyncio.wait_for(forecast.ainvoke({"city": "Boston"}), timeout=5)
        finally:
            release_lookup.set()
            if releaser is not None:
                await releaser
    lookup.assert_called_once_with(mlflow.get_tracking_uri(), "0")
    assert "Sunny in Boston" in str(result)
    if not lookup_finishes:
        # The lookup deadline bounds how long the result waits.
        assert loop.time() - started < 2
    span = mlflow.get_trace(mlflow.get_last_active_trace_id()).data.spans[0]
    assert span.attributes.get(SpanAttributeKey.MCP_SERVER_NAME) == (
        _SERVER_NAME if lookup_finishes else None
    )


@pytest.mark.parametrize("run_tracer_inline", [True, False])
@pytest.mark.asyncio
async def test_cancellation_during_workspace_lookup_exports_trace(
    mcp_adapter_class, serve_weather, run_tracer_inline
):
    mlflow.langchain.autolog(run_tracer_inline=run_tracer_inline)
    lookup_started = asyncio.Event()
    tool_started = asyncio.Event()
    release_lookup = Event()
    release_tool = Event()
    loop = asyncio.get_running_loop()

    def get_workspace(*args):
        loop.call_soon_threadsafe(lookup_started.set)
        assert release_lookup.wait(timeout=5)
        return "default"

    async def wait_for_tool():
        loop.call_soon_threadsafe(tool_started.set)
        assert await asyncio.to_thread(release_tool.wait, 5)

    with serve_weather(wait_for_tool=wait_for_tool) as mcp_url:
        with _mock_endpoints(_endpoint(url=mcp_url)) as search:
            [forecast] = await mcp_adapter_class(mcp_url).list_tools()
        search.assert_called_once_with(page_token=None)

        with mock.patch.object(_mcp, "_experiment_workspace", side_effect=get_workspace) as lookup:
            try:
                with mlflow.start_span("parent") as root:
                    task = asyncio.create_task(forecast.ainvoke({"city": "Boston"}))
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(lookup_started.wait(), tool_started.wait()), timeout=5
                        )
                    finally:
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                        assert task.cancelled()
            finally:
                release_lookup.set()
                release_tool.set()
        lookup.assert_called_once_with(mlflow.get_tracking_uri(), "0")

    trace = mlflow.get_trace(root.trace_id, flush=True)
    assert trace is not None
    [tool_span] = [span for span in trace.data.spans if span.span_type == SpanType.TOOL]
    assert tool_span.status.status_code == SpanStatusCode.ERROR
    assert SpanAttributeKey.MCP_SERVER_NAME not in tool_span.attributes
    assert all(span.end_time_ns is not None for span in trace.data.spans)


@pytest.fixture
def workspace_tracking_server(mcp_adapter_class, db_uri, tmp_path):
    from tests.tracking.integration_test_utils import _await_server_up_or_die

    # Compatibility runs set the interpreter to the MLflow environment with FastMCP 3.
    port = get_safe_port()
    uri = f"http://127.0.0.1:{port}"
    with subprocess.Popen(
        [
            os.environ.get("MLFLOW_TEST_MCP_TRACKING_PYTHON", sys.executable),
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            db_uri,
            "--artifacts-destination",
            str(tmp_path / "artifacts"),
            "--enable-workspaces",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
        ],
    ) as process:
        try:
            _await_server_up_or_die(port)
            with _use_tracking_uri(uri):
                yield
        finally:
            process.terminate()


def _register_weather_version(version):
    mlflow.genai.register_mcp_server(
        server_json={"name": _SERVER_NAME, "version": version, "description": "Weather tools"},
        status="active",
        tools=[],
    )


@pytest.mark.asyncio
async def test_adapter_links_with_in_memory_sqlite(
    mcp_adapter_class, mcp_url, monkeypatch, tmp_path
):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    with _use_tracking_uri("sqlite:///:memory:"):
        _register_weather_version("1.0.0")
        mlflow.genai.create_mcp_access_endpoint(_SERVER_NAME, mcp_url, server_version="1.0.0")
        experiment_id = mlflow.create_experiment("mcp-in-memory", artifact_location=str(tmp_path))
        mlflow.set_experiment(experiment_id=experiment_id)
        mlflow.langchain.autolog(run_tracer_inline=True)
        [forecast] = await mcp_adapter_class(mcp_url).list_tools()

        assert "Sunny in Boston" in str(await forecast.ainvoke({"city": "Boston"}))
        trace = mlflow.get_trace(mlflow.get_last_active_trace_id(), flush=True)
        [span] = trace.data.spans
        assert span.attributes[SpanAttributeKey.MCP_SERVER_NAME] == _SERVER_NAME
        assert span.attributes[SpanAttributeKey.MCP_SERVER_VERSION] == "1.0.0"
        assert span.attributes[SpanAttributeKey.MCP_TOOL_NAME] == "forecast"


@pytest.mark.parametrize(
    ("same_workspace", "tool_name"),
    [(True, "forecast"), (True, "weather_forecast"), (False, "forecast")],
    ids=["same-workspace", "renamed-tool", "different-workspace"],
)
@pytest.mark.asyncio
async def test_adapter_links_toolnode_trace(
    workspace_tracking_server, mcp_adapter_class, mcp_url, same_workspace, tool_name
):
    # With no workspace header, the tracking server selects its default.
    _register_weather_version("1.0.0")
    _register_weather_version("2.0.0")
    mlflow.genai.set_mcp_server_alias(_SERVER_NAME, "production", "1.0.0")
    mlflow.genai.create_mcp_access_endpoint(_SERVER_NAME, mcp_url, server_alias="production")

    workspace = None
    if not same_workspace:
        workspace = "other"
        mlflow.create_workspace(workspace)
        with ServerWorkspaceContext(workspace):
            _register_weather_version("1.0.0")
            mlflow.genai.create_mcp_access_endpoint(_SERVER_NAME, mcp_url, server_version="1.0.0")

    # Enabling autologging twice must not patch twice and link the span twice.
    mlflow.langchain.autolog(run_tracer_inline=True)
    mlflow.langchain.autolog(run_tracer_inline=True)
    adapter = mcp_adapter_class(target=mcp_url)
    async with adapter:
        [forecast] = await adapter.list_tools(cache_mode="refresh")
        forecast.name = tool_name
        mlflow.genai.set_mcp_server_alias(_SERVER_NAME, "production", "2.0.0")

        with ServerWorkspaceContext(workspace):
            experiment = mlflow.set_experiment("mcp-toolnode")
            graph = (
                StateGraph(MessagesState)
                .add_node("tools", ToolNode([forecast]))
                .add_edge(START, "tools")
                .compile()
            )
            with mlflow.start_span("invoke forecast") as root:
                result = await graph.ainvoke({
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[{"name": tool_name, "args": {"city": "Boston"}, "id": "1"}],
                        )
                    ]
                })
            message = result["messages"][-1]
            assert message.name == tool_name
            assert "Sunny in Boston" in str(message.content)
            trace = mlflow.get_trace(root.trace_id, flush=True)
            [span] = [s for s in trace.data.spans if s.span_type == SpanType.TOOL]
            assert span.name == tool_name
            expected = {
                SpanAttributeKey.MCP_SERVER_NAME: _SERVER_NAME,
                SpanAttributeKey.MCP_SERVER_VERSION: "1.0.0",
                SpanAttributeKey.MCP_TOOL_NAME: "forecast",
            }
            assert _mcp_attributes(span) == (expected if same_workspace else {})

            traces = mlflow.search_traces(
                locations=[experiment.experiment_id],
                filter_string=f"mcp_server = '{_SERVER_NAME}'",
                return_type="list",
                flush=True,
            )
            assert (root.trace_id in {t.info.trace_id for t in traces}) == same_workspace


@pytest.mark.parametrize("experiment", ["default", "env", "env_after_discovery"])
@pytest.mark.asyncio
async def test_adapter_links_without_set_experiment(
    workspace_tracking_server, mcp_adapter_class, mcp_url, experiment, monkeypatch
):
    _register_weather_version("1.0.0")
    mlflow.genai.create_mcp_access_endpoint(_SERVER_NAME, mcp_url, server_version="1.0.0")
    if experiment == "env":
        monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", mlflow.create_experiment("mcp-env"))
    mlflow.langchain.autolog(run_tracer_inline=True)
    async with mcp_adapter_class(mcp_url) as adapter:
        [forecast] = await adapter.list_tools()
        if experiment == "env_after_discovery":
            monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", mlflow.create_experiment("mcp-after"))
        for _ in range(3):
            assert "Sunny in Boston" in str(await forecast.ainvoke({"city": "Boston"}))
            [span] = mlflow.get_trace(mlflow.get_last_active_trace_id(), flush=True).data.spans
            assert span.attributes.get(SpanAttributeKey.MCP_SERVER_NAME) == _SERVER_NAME
            assert span.attributes.get(SpanAttributeKey.MCP_SERVER_VERSION) == "1.0.0"


@pytest.mark.asyncio
async def test_concurrent_tool_calls_share_pending_workspace_lookup(
    mcp_adapter_class, serve_weather, monkeypatch
):
    mlflow.langchain.autolog(run_tracer_inline=True)
    monkeypatch.setattr(_mcp, "_LOOKUP_TIMEOUT_SECONDS", 5)
    loop = asyncio.get_running_loop()
    tools_started = asyncio.Semaphore(0)
    release_tools = Event()
    release_lookup = Event()

    async def wait_for_tool():
        loop.call_soon_threadsafe(tools_started.release)
        assert await asyncio.to_thread(release_tools.wait, 5)

    def get_workspace(*args):
        assert release_lookup.wait(timeout=5)
        return "default"

    with serve_weather(wait_for_tool=wait_for_tool) as mcp_url:
        with _mock_endpoints(_endpoint(url=mcp_url)):
            [forecast] = await mcp_adapter_class(mcp_url).list_tools()

        with mock.patch.object(_mcp, "_experiment_workspace", side_effect=get_workspace) as lookup:
            try:
                with mlflow.start_span("parent") as root:
                    tasks = [
                        asyncio.create_task(forecast.ainvoke({"city": "Boston"})) for _ in range(4)
                    ]
                    for _ in tasks:
                        await asyncio.wait_for(tools_started.acquire(), timeout=5)
                    # Cancelling one call must not cancel the lookup the others are waiting on.
                    tasks[0].cancel()
                    release_tools.set()
                    await asyncio.sleep(0.2)
                    release_lookup.set()
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), timeout=5
                    )
            finally:
                release_tools.set()
                release_lookup.set()
            assert isinstance(results[0], asyncio.CancelledError)
            assert all("Sunny in Boston" in str(result) for result in results[1:])
            lookup.assert_called_once_with(mlflow.get_tracking_uri(), "0")

            # A finished lookup is not reused, so a moved experiment is not linked by a stale value.
            await forecast.ainvoke({"city": "Boston"})
            assert lookup.call_count == 2

    spans = [
        span
        for span in mlflow.get_trace(root.trace_id, flush=True).data.spans
        if span.span_type == SpanType.TOOL
    ]
    assert sorted(
        span.attributes.get(SpanAttributeKey.MCP_SERVER_NAME) or "" for span in spans
    ) == ["", _SERVER_NAME, _SERVER_NAME, _SERVER_NAME]


@pytest.mark.parametrize(
    ("endpoint_workspace", "trace_workspace", "linked"),
    [(None, None, True), ("other", "other", True), ("other", None, False)],
    ids=["server-default", "request-workspace", "not-in-server-default"],
)
@pytest.mark.asyncio
async def test_adapter_links_otlp_only_trace(
    workspace_tracking_server,
    mcp_adapter_class,
    mcp_url,
    endpoint_workspace,
    trace_workspace,
    linked,
    monkeypatch,
):
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from mlflow.tracing.provider import _get_tracer

    if endpoint_workspace is not None:
        mlflow.create_workspace(endpoint_workspace)
    with ServerWorkspaceContext(endpoint_workspace):
        _register_weather_version("1.0.0")
        mlflow.genai.create_mcp_access_endpoint(_SERVER_NAME, mcp_url, server_version="1.0.0")

    exporter = InMemorySpanExporter()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
    monkeypatch.setattr("mlflow.tracing.provider.get_otlp_exporter", lambda: exporter)
    mlflow.tracing.reset()
    try:
        mlflow.langchain.autolog(run_tracer_inline=True)
        with ServerWorkspaceContext(endpoint_workspace):
            [forecast] = await mcp_adapter_class(mcp_url).list_tools()
        with ServerWorkspaceContext(trace_workspace):
            assert "Sunny in Boston" in str(await forecast.ainvoke({"city": "Boston"}))
        _get_tracer(__name__).span_processor.force_flush()
    finally:
        mlflow.tracing.reset()

    [span] = exporter.get_finished_spans()
    # MCP attributes are exported as literal strings for OTLP ingestion.
    expected = {
        SpanAttributeKey.MCP_SERVER_NAME: _SERVER_NAME,
        SpanAttributeKey.MCP_SERVER_VERSION: "1.0.0",
        SpanAttributeKey.MCP_TOOL_NAME: "forecast",
    }
    assert _mcp_attributes(span) == (expected if linked else {})
