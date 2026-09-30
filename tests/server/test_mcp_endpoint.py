from contextlib import asynccontextmanager
from unittest import mock

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from starlette.testclient import TestClient

from mlflow.environment_variables import MLFLOW_SERVER_ENABLE_MCP
from mlflow.mcp.server import collect_category_tools
from mlflow.mcp.tools import SHARED_TOOLS
from mlflow.server import ARTIFACT_ROOT_ENV_VAR, BACKEND_STORE_URI_ENV_VAR, handlers
from mlflow.server.fastapi_app import create_fastapi_app
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR

_ALL_CATEGORIES = ("traces", "scorers", "experiments", "runs", "models", "deployments")


@pytest.fixture
def backend_store_env(monkeypatch, db_uri, tmp_path):
    # The host guard rejects the test client's synthetic "testserver" host on POST requests.
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, db_uri)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, str(tmp_path / "artifacts"))
    # The server caches its tracking store in a module global; start from a clean slate so the
    # endpoint resolves the store configured above.
    monkeypatch.setattr(handlers, "_tracking_store", None)


@pytest.fixture
def mcp_app(monkeypatch, backend_store_env):
    monkeypatch.setenv(MLFLOW_SERVER_ENABLE_MCP.name, "true")
    return create_fastapi_app()


@asynccontextmanager
async def _mcp_client(app, url="http://testserver/mcp"):
    # The MCP session manager is started by the server lifespan, so run it around the client.
    # Docker lookup is disabled to keep the unrelated sandbox cleanup out of these tests.
    with mock.patch("mlflow.server.fastapi_app.shutil.which", return_value=None):
        async with app.router.lifespan_context(app):
            transport = StreamableHttpTransport(
                url,
                httpx_client_factory=lambda **kwargs: httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), **kwargs
                ),
            )
            async with Client(transport) as client:
                yield client


def test_mcp_endpoint_absent_without_flag(backend_store_env):
    client = TestClient(create_fastapi_app())
    assert client.post("/mcp", json={}).status_code == 404


@pytest.mark.asyncio
async def test_mcp_endpoint_serves_exactly_the_shared_tools(mcp_app):
    async with _mcp_client(mcp_app) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools} == {tool.name for tool in SHARED_TOOLS}
    assert all(tool.outputSchema["type"] == "object" for tool in tools)
    # Tools that execute work locally stay on the stdio server.
    stdio_only = {tool.name for tool in collect_category_tools(_ALL_CATEGORIES)} - {
        tool.name for tool in tools
    }
    assert {"evaluate_traces", "serve_model", "predict_with_model", "create_deployment"} <= (
        stdio_only
    )


@pytest.mark.asyncio
async def test_mcp_tool_round_trips_through_backend_store(mcp_app):
    store = handlers._get_tracking_store()
    store.create_experiment("created-in-store")

    async with _mcp_client(mcp_app) as client:
        result = await client.call_tool("search_experiments", {})
        names = {e["name"] for e in result.structured_content["experiments"]}
        assert "created-in-store" in names

        result = await client.call_tool("create_experiment", {"experiment_name": "created-via-mcp"})

    experiment = store.get_experiment_by_name("created-via-mcp")
    assert result.structured_content == {
        "experiment_id": experiment.experiment_id,
        "name": "created-via-mcp",
    }


@pytest.mark.asyncio
async def test_mcp_endpoint_honors_static_prefix(monkeypatch, backend_store_env):
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/myprefix")
    monkeypatch.setenv(MLFLOW_SERVER_ENABLE_MCP.name, "true")
    app = create_fastapi_app()

    async with _mcp_client(app, url="http://testserver/myprefix/mcp") as client:
        assert await client.list_tools()

    assert TestClient(app).post("/mcp", json={}).status_code == 404
