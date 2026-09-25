from contextlib import asynccontextmanager
from unittest import mock

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from starlette.testclient import TestClient

from mlflow.environment_variables import MLFLOW_SERVER_ENABLE_MCP
from mlflow.mcp.server import collect_category_tools
from mlflow.mcp.server_app import LOCAL_EXECUTION_TOOLS, SERVER_MCP_TOOL_CATEGORIES
from mlflow.server import ARTIFACT_ROOT_ENV_VAR, BACKEND_STORE_URI_ENV_VAR, handlers
from mlflow.server.fastapi_app import create_fastapi_app
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR

_ML_ONLY_TOOLS = {"serve_model", "predict_with_model", "create_deployment", "list_deployments"}


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
async def test_mcp_endpoint_lists_genai_tools_only(mcp_app):
    async with _mcp_client(mcp_app) as client:
        names = {tool.name for tool in await client.list_tools()}

    assert {"search_experiments", "list_runs", "search_traces", "list_scorers"} <= names
    assert names.isdisjoint(_ML_ONLY_TOOLS)
    # Tools that execute work locally stay on the stdio server.
    assert names.isdisjoint(LOCAL_EXECUTION_TOOLS)
    assert LOCAL_EXECUTION_TOOLS <= {
        tool.name for tool in collect_category_tools(SERVER_MCP_TOOL_CATEGORIES)
    }


@pytest.mark.asyncio
async def test_mcp_tool_round_trips_through_backend_store(mcp_app):
    store = handlers._get_tracking_store()
    store.create_experiment("created-in-store")

    async with _mcp_client(mcp_app) as client:
        result = await client.call_tool("search_experiments", {})
        assert "created-in-store" in result.content[0].text

        await client.call_tool("create_experiment", {"experiment_name": "created-via-mcp"})

    assert store.get_experiment_by_name("created-via-mcp") is not None


@pytest.mark.asyncio
async def test_mcp_endpoint_honors_static_prefix(monkeypatch, backend_store_env):
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/myprefix")
    monkeypatch.setenv(MLFLOW_SERVER_ENABLE_MCP.name, "true")
    app = create_fastapi_app()

    async with _mcp_client(app, url="http://testserver/myprefix/mcp") as client:
        assert await client.list_tools()

    assert TestClient(app).post("/mcp", json={}).status_code == 404
