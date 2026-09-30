import pytest

from mlflow.environment_variables import MLFLOW_SERVER_ENABLE_MCP
from mlflow.mcp.tools import SHARED_TOOLS
from mlflow.server import ARTIFACT_ROOT_ENV_VAR, BACKEND_STORE_URI_ENV_VAR, handlers
from mlflow.server.fastapi_app import create_fastapi_app

from tests.mcp.helpers import SCENARIOS, http_client, normalize, stdio_client


def test_scenarios_cover_every_shared_tool():
    assert sorted(s.tool for s in SCENARIOS) == sorted(t.name for t in SHARED_TOOLS)


@pytest.fixture
def mcp_app(monkeypatch, db_uri, tmp_path):
    # The host guard rejects the test client's synthetic "testserver" host on POST requests.
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, db_uri)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, str(tmp_path / "artifacts"))
    monkeypatch.setenv(MLFLOW_SERVER_ENABLE_MCP.name, "true")
    monkeypatch.setattr(handlers, "_tracking_store", None)
    return create_fastapi_app()


@pytest.mark.asyncio
async def test_every_shared_tool_returns_the_same_structured_content_on_both_transports(
    mcp_app, db_uri
):
    # Both transports serve the same sqlite store: stdio through its tracking URI, /mcp as the
    # server's backend store.
    async with stdio_client(db_uri) as stdio, http_client(mcp_app) as http:
        stdio_tools = {tool.name: tool for tool in await stdio.list_tools()}
        http_tools = {tool.name: tool for tool in await http.list_tools()}
        for name in http_tools:
            assert stdio_tools[name].inputSchema == http_tools[name].inputSchema
            assert stdio_tools[name].outputSchema == http_tools[name].outputSchema

        for scenario in SCENARIOS:
            if scenario.read_only:
                stdio_args = http_args = scenario.setup("shared")
            else:
                stdio_args = scenario.setup("stdio")
                http_args = scenario.setup("http")

            stdio_result = await stdio.call_tool(scenario.tool, stdio_args)
            http_result = await http.call_tool(scenario.tool, http_args)

            assert stdio_result.structured_content is not None
            if scenario.read_only:
                assert stdio_result.structured_content == http_result.structured_content
            else:
                assert normalize(
                    stdio_result.structured_content, stdio_args, scenario.generated
                ) == normalize(http_result.structured_content, http_args, scenario.generated)
