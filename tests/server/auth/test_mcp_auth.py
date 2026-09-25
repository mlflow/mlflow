# Authentication and per-tool authorization of the Streamable HTTP MCP endpoint under the
# basic-auth app. The server is spawned the same way as the other FastAPI auth tests; a
# ``NO_PERMISSIONS`` default makes every grant explicit so denials are meaningful.

import time
from pathlib import Path

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError

import mlflow
from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_FLASK_SERVER_SECRET_KEY, MLFLOW_SERVER_ENABLE_MCP
from mlflow.exceptions import MlflowException
from mlflow.mcp.server import collect_category_tools
from mlflow.mcp.server_app import LOCAL_EXECUTION_TOOLS, SERVER_MCP_TOOL_CATEGORIES
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, ErrorCode
from mlflow.server import auth as auth_module
from mlflow.server.auth.mcp_tools import (
    MCP_TOOL_RULES,
    authorize_mcp_tool_call,
    check_mcp_tool_coverage,
)
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR
from mlflow.utils.os import is_windows

from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    User,
    create_user,
    grant_role_permission,
)
from tests.tracking.integration_test_utils import _init_server

_INITIALIZE_REQUEST = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
_MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def _write_auth_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "basic_auth.ini"
    config_path.write_text(
        "[mlflow]\n"
        "default_permission = NO_PERMISSIONS\n"
        f"database_uri = sqlite:///{tmp_path / 'basic_auth.db'}\n"
        f"admin_username = {ADMIN_USERNAME}\n"
        f"admin_password = {ADMIN_PASSWORD}\n"
        "authorization_function = mlflow.server.auth:authenticate_request_basic_auth\n"
    )
    return config_path


def _backend_uri(tmp_path: Path) -> str:
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    return ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]


@pytest.fixture
def mcp_server(request, tmp_path):
    extra_env = {
        MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key",
        "MLFLOW_AUTH_CONFIG_PATH": str(_write_auth_config(tmp_path)),
        "_MLFLOW_SGI_NAME": "uvicorn",
        MLFLOW_SERVER_ENABLE_MCP.name: "true",
        **getattr(request, "param", {}),
    }
    with _init_server(
        backend_uri=_backend_uri(tmp_path),
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        extra_env=extra_env,
        app="mlflow.server.auth:create_app",
        server_type="fastapi",
    ) as url:
        yield url


@pytest.fixture
def unauthenticated_mcp_server(tmp_path):
    with _init_server(
        backend_uri=_backend_uri(tmp_path),
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        extra_env={MLFLOW_SERVER_ENABLE_MCP.name: "true"},
        server_type="fastapi",
    ) as url:
        yield url


def _mcp_client(url: str, credentials: tuple[str, str] | None, path: str = "/mcp") -> Client:
    auth = httpx.BasicAuth(*credentials) if credentials else None
    return Client(StreamableHttpTransport(f"{url}{path}", auth=auth))


async def _call(url: str, credentials: tuple[str, str] | None, tool: str, **arguments) -> str:
    async with _mcp_client(url, credentials) as client:
        result = await client.call_tool(tool, arguments)
    return result.content[0].text


def _admin_client(url: str, monkeypatch) -> MlflowClient:
    User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch).__enter__()
    return MlflowClient(url)


def _experiments(url: str, monkeypatch, names: list[str]) -> list[str]:
    client = _admin_client(url, monkeypatch)
    return [client.create_experiment(name) for name in names]


def _reader(url: str, *experiment_ids: str, permission: str = "READ") -> tuple[str, str]:
    username, password = create_user(url)
    for experiment_id in experiment_ids:
        grant_role_permission(url, username, "experiment", experiment_id, permission)
    return username, password


def _log_trace(url: str, monkeypatch, experiment_id: str) -> str:
    _admin_client(url, monkeypatch)
    mlflow.set_tracking_uri(url)
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_span("span") as span:
        pass
    mlflow.flush_trace_async_logging()
    return span.trace_id


ADMIN = (ADMIN_USERNAME, ADMIN_PASSWORD)


# --------------------------------------------------------------------------- authentication


def test_missing_or_wrong_credentials_get_the_rest_basic_auth_challenge(mcp_server):
    for auth in (None, httpx.BasicAuth("nobody", "wrong")):
        response = httpx.post(
            f"{mcp_server}/mcp", json=_INITIALIZE_REQUEST, headers=_MCP_HEADERS, auth=auth
        )
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == 'Basic realm="mlflow"'
        assert "You are not authenticated" in response.text


@pytest.mark.asyncio
async def test_endpoint_is_open_when_auth_app_is_not_active(unauthenticated_mcp_server):
    text = await _call(unauthenticated_mcp_server, None, "search_experiments")
    assert "Default" in text


# --------------------------------------------------------------------------- authorization


@pytest.mark.asyncio
async def test_reader_is_scoped_to_granted_experiment(mcp_server, monkeypatch):
    exp_a, exp_b = _experiments(mcp_server, monkeypatch, ["exp-a", "exp-b"])
    reader = _reader(mcp_server, exp_a)

    assert "exp-a" in await _call(mcp_server, reader, "get_experiment", experiment_id=exp_a)
    # An empty trace search still exercises the experiment read gate.
    await _call(mcp_server, reader, "search_traces", experiment_id=exp_a)

    for tool in ("get_experiment", "search_traces"):
        with pytest.raises(ToolError, match="^Permission denied$"):
            await _call(mcp_server, reader, tool, experiment_id=exp_b)


@pytest.mark.asyncio
async def test_reader_cannot_delete_traces_but_manager_and_admin_can(mcp_server, monkeypatch):
    (exp_a,) = _experiments(mcp_server, monkeypatch, ["exp-a"])
    reader = _reader(mcp_server, exp_a)
    manager = _reader(mcp_server, exp_a, permission="MANAGE")
    now = int(time.time() * 1000)

    with pytest.raises(ToolError, match="^Permission denied$"):
        await _call(
            mcp_server, reader, "delete_traces", experiment_id=exp_a, max_timestamp_millis=now
        )

    for credentials in (manager, ADMIN):
        text = await _call(
            mcp_server, credentials, "delete_traces", experiment_id=exp_a, max_timestamp_millis=now
        )
        assert "Deleted 0 trace" in text


@pytest.mark.asyncio
async def test_admin_passes_every_check(mcp_server, monkeypatch):
    (exp_b,) = _experiments(mcp_server, monkeypatch, ["exp-b"])
    assert "exp-b" in await _call(mcp_server, ADMIN, "get_experiment", experiment_id=exp_b)
    await _call(mcp_server, ADMIN, "rename_experiment", experiment_id=exp_b, new_name="exp-b2")
    assert "exp-b2" in await _call(mcp_server, ADMIN, "get_experiment", experiment_id=exp_b)


@pytest.mark.asyncio
async def test_run_tools_resolve_the_run_experiment(mcp_server, monkeypatch):
    exp_a, exp_b = _experiments(mcp_server, monkeypatch, ["exp-a", "exp-b"])
    client = _admin_client(mcp_server, monkeypatch)
    run_a = client.create_run(exp_a).info.run_id
    run_b = client.create_run(exp_b).info.run_id
    reader = _reader(mcp_server, exp_a)

    assert run_a in await _call(mcp_server, reader, "describe_run", run_id=run_a)
    with pytest.raises(ToolError, match="^Permission denied$"):
        await _call(mcp_server, reader, "describe_run", run_id=run_b)
    # A missing run denies rather than surfacing a not-found error.
    with pytest.raises(ToolError, match="^Permission denied$"):
        await _call(mcp_server, reader, "describe_run", run_id="no-such-run")


@pytest.mark.asyncio
async def test_trace_tools_resolve_the_trace_experiment(mcp_server, monkeypatch):
    exp_a, exp_b = _experiments(mcp_server, monkeypatch, ["exp-a", "exp-b"])
    trace_a = _log_trace(mcp_server, monkeypatch, exp_a)
    trace_b = _log_trace(mcp_server, monkeypatch, exp_b)
    reader = _reader(mcp_server, exp_a)

    assert trace_a in await _call(mcp_server, reader, "get_trace", trace_id=trace_a)
    with pytest.raises(ToolError, match="^Permission denied$"):
        await _call(mcp_server, reader, "get_trace", trace_id=trace_b)
    with pytest.raises(ToolError, match="^Permission denied$"):
        await _call(mcp_server, reader, "get_trace", trace_id="tr-no-such-trace")
    # Tagging needs update rights: READ on the trace's experiment is not enough.
    with pytest.raises(ToolError, match="^Permission denied$"):
        await _call(mcp_server, reader, "set_trace_tag", trace_id=trace_a, key="k", value="v")


@pytest.mark.asyncio
async def test_unscoped_search_experiments_fills_the_page_with_readable_rows(
    mcp_server, monkeypatch
):
    # Default ordering is newest first, so the two readable experiments (created first) sit on
    # the last store pages; a page size of 2 must be refilled across unreadable pages.
    ids = _experiments(mcp_server, monkeypatch, [f"exp-{i}" for i in range(5)])
    reader = _reader(mcp_server, ids[0], ids[1])

    text = await _call(mcp_server, reader, "search_experiments", max_results=2)
    assert "exp-0" in text
    assert "exp-1" in text
    assert all(name not in text for name in ("exp-2", "exp-3", "exp-4", "Default"))

    unlimited = await _call(mcp_server, reader, "search_experiments")
    assert unlimited == text

    assert "exp-4" in await _call(mcp_server, ADMIN, "search_experiments", max_results=2)


@pytest.mark.asyncio
async def test_reader_can_search_experiments_with_no_grants_and_sees_nothing(
    mcp_server, monkeypatch
):
    _experiments(mcp_server, monkeypatch, ["exp-a"])
    nobody = create_user(mcp_server)
    text = await _call(mcp_server, nobody, "search_experiments")
    assert "exp-a" not in text
    assert "Default" not in text


@pytest.mark.parametrize("mcp_server", [{"MLFLOW_BASIC_AUTH_FAIL_CLOSED": "true"}], indirect=True)
@pytest.mark.asyncio
async def test_fail_closed_mode_keeps_the_authenticated_endpoint_reachable(mcp_server, monkeypatch):
    (exp_a,) = _experiments(mcp_server, monkeypatch, ["exp-a"])
    reader = _reader(mcp_server, exp_a)
    assert "exp-a" in await _call(mcp_server, reader, "get_experiment", experiment_id=exp_a)


@pytest.mark.parametrize("mcp_server", [{STATIC_PREFIX_ENV_VAR: "/myprefix"}], indirect=True)
@pytest.mark.asyncio
async def test_static_prefix_route_is_authenticated_and_authorized(mcp_server, monkeypatch):
    prefixed = f"{mcp_server}/myprefix/mcp"
    response = httpx.post(prefixed, json=_INITIALIZE_REQUEST, headers=_MCP_HEADERS)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == 'Basic realm="mlflow"'

    exp_a, exp_b = _experiments(mcp_server + "/myprefix", monkeypatch, ["exp-a", "exp-b"])
    reader = _reader(mcp_server + "/myprefix", exp_a)
    async with _mcp_client(mcp_server, reader, path="/myprefix/mcp") as client:
        result = await client.call_tool("get_experiment", {"experiment_id": exp_a})
        assert "exp-a" in result.content[0].text
        with pytest.raises(ToolError, match="^Permission denied$"):
            await client.call_tool("get_experiment", {"experiment_id": exp_b})


def test_find_fastapi_validator_resolves_the_mcp_path_under_a_static_prefix(monkeypatch):
    monkeypatch.delenv(STATIC_PREFIX_ENV_VAR, raising=False)
    assert auth_module._find_fastapi_validator("/mcp", "POST") is not None
    assert auth_module._find_fastapi_validator("/myprefix/mcp", "POST") is None

    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/myprefix")
    assert auth_module._find_fastapi_validator("/myprefix/mcp", "POST") is not None
    # Like the other native routes, the unprefixed root still resolves a validator: nothing
    # serves it once a prefix is configured, and this errs toward requiring auth.
    assert auth_module._find_fastapi_validator("/mcp", "POST") is not None
    assert auth_module._find_fastapi_validator("/myprefix/mcp/other", "POST") is None


# --------------------------------------------------------------------------- rule table


def test_every_served_tool_has_a_rule_and_nothing_else():
    served = {
        tool.name
        for tool in collect_category_tools(SERVER_MCP_TOOL_CATEGORIES)
        if tool.name not in LOCAL_EXECUTION_TOOLS
    }
    assert served == set(MCP_TOOL_RULES)
    check_mcp_tool_coverage(served)


def test_startup_coverage_check_rejects_an_unlisted_tool():
    with pytest.raises(MlflowException, match=r"\['brand_new_tool'\]"):
        check_mcp_tool_coverage(["get_experiment", "brand_new_tool"])


@pytest.mark.parametrize(
    ("tool", "username"),
    [
        ("get_experiment", None),
        ("brand_new_tool", "alice"),
    ],
)
def test_authorize_denies_without_identity_or_rule(tool: str, username: str | None):
    with pytest.raises(MlflowException, match="Permission denied") as exc_info:
        authorize_mcp_tool_call(tool, username, {"experiment_id": "0"})
    assert exc_info.value.error_code == ErrorCode.Name(PERMISSION_DENIED)
