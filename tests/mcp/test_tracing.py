import asyncio
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import uvicorn
from fastapi.testclient import TestClient
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from mlflow.entities import Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.mcp.tracing import instrument_fastmcp
from mlflow.server.fastapi_app import create_fastapi_app
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH
from mlflow.utils import workspace_context
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME

_MCP_ATTRIBUTES = (
    SpanAttributeKey.MCP_SERVER_NAME,
    SpanAttributeKey.MCP_SERVER_VERSION,
    SpanAttributeKey.MCP_TOOL_NAME,
)


@pytest.fixture
def otel_provider(monkeypatch):
    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", None)
    monkeypatch.setattr(otel_trace._TRACER_PROVIDER_SET_ONCE, "_done", False)

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    yield provider, exporter
    provider.shutdown()


@contextmanager
def _running_server(server: FastMCP) -> Iterator[str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        uvicorn_server = uvicorn.Server(
            uvicorn.Config(
                server.http_app(path="/mcp"),
                host="127.0.0.1",
                port=port,
                lifespan="on",
                ws="none",
                timeout_graceful_shutdown=2,
            )
        )
        thread = threading.Thread(
            target=uvicorn_server.run,
            kwargs={"sockets": [sock]},
            name="fastmcp-tracing-test",
            daemon=True,
        )
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not uvicorn_server.started:
                if not thread.is_alive() or time.monotonic() >= deadline:
                    pytest.fail("FastMCP test server did not become ready")
                time.sleep(0.05)
            yield f"http://127.0.0.1:{port}/mcp"
        finally:
            uvicorn_server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()


def _tool_server() -> FastMCP:
    server = FastMCP("tracing-test")

    @server.tool
    def forecast(city: str) -> str:
        return f"Forecast for {city}"

    return server


def _tool_spans(exporter: InMemorySpanExporter):
    return [
        span
        for span in exporter.get_finished_spans()
        if span.kind == SpanKind.SERVER and span.name.startswith("tools/call ")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, "1.2.3"])
async def test_instrument_fastmcp_stamps_native_tool_span(otel_provider, version):
    _, exporter = otel_provider
    server = _tool_server()
    instrument_fastmcp(
        server,
        name="com.example/weather",
        version=version,
    )

    tracer = otel_trace.get_tracer("weather-agent")
    async with Client(server) as client:
        with tracer.start_as_current_span("agent.run"):
            await client.list_tools()
            await client.call_tool("forecast", {"city": "Seattle"})

    tool_spans = _tool_spans(exporter)
    assert len(tool_spans) == 1
    server_span = tool_spans[0]
    assert server_span.name == "tools/call forecast"
    expected = {
        SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
        SpanAttributeKey.MCP_TOOL_NAME: "forecast",
    }
    if version is not None:
        expected[SpanAttributeKey.MCP_SERVER_VERSION] = version
    assert {key: server_span.attributes.get(key) for key in expected} == expected
    assert (SpanAttributeKey.MCP_SERVER_VERSION in server_span.attributes) == (version is not None)

    spans = exporter.get_finished_spans()
    assert any(span.name == "agent.run" for span in spans)
    assert any(span.kind == SpanKind.CLIENT for span in spans)
    for span in spans:
        if span is not server_span:
            assert not set(_MCP_ATTRIBUTES).intersection(span.attributes)


@pytest.mark.asyncio
async def test_fastmcp_client_propagates_trace_to_server_span(otel_provider):
    _, exporter = otel_provider
    server = _tool_server()
    instrument_fastmcp(
        server,
        name="com.example/weather",
        version="1.2.3",
    )

    with _running_server(server) as server_url:
        tracer = otel_trace.get_tracer("weather-agent")
        async with Client(server_url) as client:
            with tracer.start_as_current_span("agent.run") as agent_span:
                await client.call_tool("forecast", {"city": "Seattle"})

    spans = exporter.get_finished_spans()
    agent_span = next(span for span in spans if span.name == "agent.run")
    client_span = next(
        span
        for span in spans
        if span.kind == SpanKind.CLIENT and span.name == "tools/call forecast"
    )
    server_span = _tool_spans(exporter)[0]

    assert client_span.context.trace_id == agent_span.context.trace_id
    assert client_span.parent.span_id == agent_span.context.span_id
    assert server_span.context.trace_id == agent_span.context.trace_id
    assert server_span.parent.span_id == client_span.context.span_id
    assert server_span.parent.is_remote


@pytest.mark.asyncio
async def test_request_without_propagated_context_starts_root_server_trace(otel_provider):
    _, exporter = otel_provider
    server = _tool_server()
    instrument_fastmcp(server, name="com.example/weather")

    with _running_server(server) as server_url:
        async with Client(server_url) as client:
            # Use the underlying MCP session so FastMCP's client span and automatic
            # trace-context injection do not run for this request.
            await client.session.call_tool(
                name="forecast",
                arguments={"city": "Seattle"},
                meta=None,
            )

    server_span = _tool_spans(exporter)[0]
    assert server_span.parent is None
    assert server_span.context.trace_id != 0
    assert server_span.attributes[SpanAttributeKey.MCP_SERVER_NAME] == "com.example/weather"
    assert server_span.attributes[SpanAttributeKey.MCP_TOOL_NAME] == "forecast"
    assert SpanAttributeKey.MCP_SERVER_VERSION not in server_span.attributes


def test_instrument_fastmcp_is_idempotent_and_rejects_conflicts(otel_provider):
    server = _tool_server()
    initial_middleware_count = len(server.middleware)

    instrument_fastmcp(server, name="com.example/weather")
    instrument_fastmcp(server, name="com.example/weather")

    assert len(server.middleware) == initial_middleware_count + 1
    with pytest.raises(MlflowException, match="already instrumented"):
        instrument_fastmcp(server, name="com.example/other")
    assert len(server.middleware) == initial_middleware_count + 1


def test_instrument_fastmcp_requires_sdk_global_provider(monkeypatch):
    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", None)
    monkeypatch.setattr(otel_trace._TRACER_PROVIDER_SET_ONCE, "_done", False)
    server = _tool_server()
    initial_middleware_count = len(server.middleware)

    with pytest.raises(MlflowException, match="initialized OpenTelemetry SDK TracerProvider"):
        instrument_fastmcp(server, name="com.example/weather")
    assert len(server.middleware) == initial_middleware_count


@pytest.mark.asyncio
async def test_concurrent_fastmcp_servers_share_provider_without_cross_attribution(otel_provider):
    _, exporter = otel_provider
    weather_server = FastMCP("weather")
    maps_server = FastMCP("maps")

    @weather_server.tool
    async def forecast(city: str) -> str:
        await asyncio.sleep(0.01)
        return f"Forecast for {city}"

    @maps_server.tool
    async def forecast(city: str) -> str:
        await asyncio.sleep(0.01)
        return f"Map for {city}"

    instrument_fastmcp(weather_server, name="com.example/weather")
    instrument_fastmcp(maps_server, name="com.example/maps")

    await asyncio.gather(
        weather_server.call_tool("forecast", {"city": "Seattle"}),
        maps_server.call_tool("forecast", {"city": "Seattle"}),
    )

    assert {
        (
            span.attributes[SpanAttributeKey.MCP_SERVER_NAME],
            span.attributes[SpanAttributeKey.MCP_TOOL_NAME],
        )
        for span in _tool_spans(exporter)
    } == {
        ("com.example/weather", "forecast"),
        ("com.example/maps", "forecast"),
    }


@pytest.mark.asyncio
async def test_tool_failure_preserves_fastmcp_error_and_cleans_request_context(otel_provider):
    provider, exporter = otel_provider
    server = FastMCP("tracing-test")

    @server.tool
    def fail() -> str:
        raise ValueError("tool failed")

    instrument_fastmcp(server, name="com.example/failure")

    with pytest.raises(ToolError, match="tool failed"):
        await server.call_tool("fail")

    server_span = _tool_spans(exporter)[0]
    assert server_span.attributes[SpanAttributeKey.MCP_SERVER_NAME] == "com.example/failure"
    assert server_span.attributes[SpanAttributeKey.MCP_TOOL_NAME] == "fail"
    assert server_span.status.status_code == StatusCode.ERROR

    tracer = provider.get_tracer("fastmcp")
    with tracer.start_as_current_span("tools/call unrelated", kind=SpanKind.SERVER):
        pass
    unrelated_span = next(
        span for span in exporter.get_finished_spans() if span.name == "tools/call unrelated"
    )
    assert not set(_MCP_ATTRIBUTES).intersection(unrelated_span.attributes)


@pytest.mark.asyncio
async def test_fastmcp_spans_can_be_ingested_and_searched_in_workspace(
    monkeypatch, db_uri, tmp_path, otel_provider
):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    store = WorkspaceAwareSqlAlchemyStore(db_uri, tmp_path.as_uri())
    with workspace_context.WorkspaceContext("team-a"):
        experiment_id = store.create_experiment("fastmcp-otel-ingestion")
    monkeypatch.setattr("mlflow.server.otel_api._get_tracking_store", lambda: store)
    monkeypatch.setattr("mlflow.server.handlers._tracking_store", store)
    monkeypatch.setattr(
        "mlflow.server.fastapi_app.resolve_workspace_for_request_if_enabled",
        lambda _path, header_workspace: Workspace(name=header_workspace),
    )

    provider, exporter = otel_provider
    server = _tool_server()
    instrument_fastmcp(
        server,
        name="com.example/weather",
        version="1.2.3",
    )
    with _running_server(server) as server_url:
        async with Client(server_url) as client:
            with provider.get_tracer("weather-agent").start_as_current_span("agent.run"):
                await client.call_tool("forecast", {"city": "Seattle"})

    spans = exporter.get_finished_spans()
    server_span = next(span for span in spans if span.kind == SpanKind.SERVER)
    trace_id = f"tr-{server_span.context.trace_id:032x}"
    request = encode_spans(spans)
    tracking_client = TestClient(create_fastapi_app(), base_url="http://localhost")
    response = tracking_client.post(
        OTLP_TRACES_PATH,
        content=request.SerializeToString(),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": experiment_id,
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )
    assert response.status_code == 200

    search_response = tracking_client.post(
        "/api/3.0/mlflow/traces/search",
        json={
            "locations": [{"mlflow_experiment": {"experiment_id": experiment_id}}],
            "filter": "mcp_server = 'com.example/weather' AND mcp_server.version = '1.2.3'",
            "max_results": 10,
        },
        headers={WORKSPACE_HEADER_NAME: "team-a"},
    )
    assert search_response.status_code == 200
    assert [trace["trace_id"] for trace in search_response.json()["traces"]] == [trace_id]
