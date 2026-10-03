# Authentication and per-tool authorization of the Streamable HTTP MCP endpoint under the
# basic-auth app. The server is spawned the same way as the other FastAPI auth tests; a
# ``NO_PERMISSIONS`` default makes every grant explicit so denials are meaningful.

import inspect
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import requests
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Experiment
from mlflow.environment_variables import (
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_RBAC_SEED_DEFAULT_ROLES,
    MLFLOW_SERVER_ENABLE_MCP,
    MLFLOW_WORKSPACE_STORE_URI,
)
from mlflow.exceptions import MlflowException
from mlflow.mcp.tools import SHARED_TOOLS
from mlflow.mcp.tools.experiments import search_experiments
from mlflow.mcp.tools.scorers import list_scorers
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, ErrorCode
from mlflow.server import auth as auth_module
from mlflow.server.auth import mcp_tools
from mlflow.server.auth.mcp_tools import (
    MCP_TOOL_RULES,
    SEARCH_READABLE_EXPERIMENTS_MAX_STORE_PAGES,
    authorize_mcp_tool_call,
    check_mcp_tool_coverage,
    list_readable_scorers,
    search_readable_experiments,
)
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR
from mlflow.store.entities.paged_list import PagedList
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.utils.os import is_windows
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME

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


def _mcp_client(
    url: str,
    credentials: tuple[str, str] | None,
    path: str = "/mcp",
    workspace: str | None = None,
) -> Client:
    auth = httpx.BasicAuth(*credentials) if credentials else None
    headers = {WORKSPACE_HEADER_NAME: workspace} if workspace else None
    return Client(StreamableHttpTransport(f"{url}{path}", auth=auth, headers=headers))


async def _call(
    url: str,
    credentials: tuple[str, str] | None,
    tool: str,
    workspace: str | None = None,
    **arguments,
) -> dict[str, Any]:
    async with _mcp_client(url, credentials, workspace=workspace) as client:
        result = await client.call_tool(tool, arguments)
    return result.structured_content


def _names(page: dict[str, Any]) -> list[str]:
    return [experiment["name"] for experiment in page["experiments"]]


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
    page = await _call(unauthenticated_mcp_server, None, "search_experiments")
    assert _names(page) == ["Default"]


# --------------------------------------------------------------------------- authorization


@pytest.mark.asyncio
async def test_reader_is_scoped_to_granted_experiment(mcp_server, monkeypatch):
    exp_a, exp_b = _experiments(mcp_server, monkeypatch, ["exp-a", "exp-b"])
    reader = _reader(mcp_server, exp_a)

    experiment = await _call(mcp_server, reader, "get_experiment", experiment_id=exp_a)
    assert experiment["name"] == "exp-a"
    # An empty trace search still exercises the experiment read gate.
    assert await _call(mcp_server, reader, "search_traces", experiment_id=exp_a) == {
        "traces": [],
        "next_page_token": None,
    }

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
        result = await _call(
            mcp_server, credentials, "delete_traces", experiment_id=exp_a, max_timestamp_millis=now
        )
        assert result == {"experiment_id": exp_a, "deleted_count": 0}


@pytest.mark.asyncio
async def test_admin_passes_every_check(mcp_server, monkeypatch):
    (exp_b,) = _experiments(mcp_server, monkeypatch, ["exp-b"])
    experiment = await _call(mcp_server, ADMIN, "get_experiment", experiment_id=exp_b)
    assert experiment["name"] == "exp-b"
    await _call(mcp_server, ADMIN, "rename_experiment", experiment_id=exp_b, new_name="exp-b2")
    experiment = await _call(mcp_server, ADMIN, "get_experiment", experiment_id=exp_b)
    assert experiment["name"] == "exp-b2"


@pytest.mark.asyncio
async def test_run_tools_resolve_the_run_experiment(mcp_server, monkeypatch):
    exp_a, exp_b = _experiments(mcp_server, monkeypatch, ["exp-a", "exp-b"])
    client = _admin_client(mcp_server, monkeypatch)
    run_a = client.create_run(exp_a).info.run_id
    run_b = client.create_run(exp_b).info.run_id
    reader = _reader(mcp_server, exp_a)

    assert (await _call(mcp_server, reader, "describe_run", run_id=run_a))["run_id"] == run_a
    with pytest.raises(ToolError, match="^Permission denied$"):
        await _call(mcp_server, reader, "describe_run", run_id=run_b)
    # A missing run denies rather than surfacing a not-found error.
    with pytest.raises(ToolError, match="^Permission denied$"):
        await _call(mcp_server, reader, "describe_run", run_id="no-such-run")


@pytest.mark.asyncio
async def test_create_run_needs_read_on_the_parent_run(mcp_server, monkeypatch):
    exp_a, exp_b = _experiments(mcp_server, monkeypatch, ["exp-a", "exp-b"])
    parent = _admin_client(mcp_server, monkeypatch).create_run(exp_a).info.run_id
    editor = _reader(mcp_server, exp_b, permission="EDIT")

    # EDIT on the destination does not allow nesting under a run the caller cannot read, and a
    # missing parent denies the same way rather than disclosing that it does not exist.
    for parent_run_id in (parent, "no-such-run"):
        with pytest.raises(ToolError, match="^Permission denied$"):
            await _call(
                mcp_server,
                editor,
                "create_run",
                experiment_id=exp_b,
                parent_run_id=parent_run_id,
            )

    grant_role_permission(mcp_server, editor[0], "experiment", exp_a, "READ")
    run = await _call(mcp_server, editor, "create_run", experiment_id=exp_b, parent_run_id=parent)
    assert run["experiment_id"] == exp_b
    assert run["status"] == "FINISHED"
    child = await _call(mcp_server, editor, "describe_run", run_id=run["run_id"])
    assert child["tags"][MLFLOW_PARENT_RUN_ID] == parent
    assert child["status"] == "FINISHED"

    # READ on the parent's experiment still does not allow creating runs in it.
    with pytest.raises(ToolError, match="^Permission denied$"):
        await _call(mcp_server, editor, "create_run", experiment_id=exp_a, parent_run_id=parent)


@pytest.mark.asyncio
async def test_trace_tools_resolve_the_trace_experiment(mcp_server, monkeypatch):
    exp_a, exp_b = _experiments(mcp_server, monkeypatch, ["exp-a", "exp-b"])
    trace_a = _log_trace(mcp_server, monkeypatch, exp_a)
    trace_b = _log_trace(mcp_server, monkeypatch, exp_b)
    reader = _reader(mcp_server, exp_a)

    trace = (await _call(mcp_server, reader, "get_trace", trace_id=trace_a))["trace"]
    assert trace["info"]["trace_id"] == trace_a
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

    page = await _call(mcp_server, reader, "search_experiments", max_results=2)
    assert sorted(_names(page)) == ["exp-0", "exp-1"]
    # The page is full; the token resumes after the rows it consumed, and nothing is left.
    assert page["next_page_token"] is not None
    rest = await _call(
        mcp_server, reader, "search_experiments", max_results=2, page_token=page["next_page_token"]
    )
    assert rest == {"experiments": [], "next_page_token": None}

    unlimited = await _call(mcp_server, reader, "search_experiments")
    assert unlimited["experiments"] == page["experiments"]
    assert unlimited["next_page_token"] is None

    admin_page = await _call(mcp_server, ADMIN, "search_experiments", max_results=2)
    assert _names(admin_page) == ["exp-4", "exp-3"]


@pytest.mark.asyncio
async def test_admin_search_experiments_pages_with_the_store_token(mcp_server, monkeypatch):
    _experiments(mcp_server, monkeypatch, ["exp-a", "exp-b"])
    first = await _call(mcp_server, ADMIN, "search_experiments", max_results=2)
    assert _names(first) == ["exp-b", "exp-a"]
    second = await _call(
        mcp_server, ADMIN, "search_experiments", max_results=2, page_token=first["next_page_token"]
    )
    assert _names(second) == ["Default"]
    assert second["next_page_token"] is None


class _OffsetPagedClient:
    """
    Stand-in for ``MlflowClient`` over a fixed experiment list, paged with offset tokens like the
    SQL and file stores.
    """

    def __init__(self, experiment_ids: list[str]):
        self.experiment_ids = experiment_ids
        self.page_sizes: list[int] = []

    def search_experiments(self, view_type, max_results, filter_string, order_by, page_token):
        self.page_sizes.append(max_results)
        start = int(page_token or 0)
        end = start + max_results
        experiments = [
            Experiment(experiment_id, experiment_id, f"file:///{experiment_id}", "active")
            for experiment_id in self.experiment_ids[start:end]
        ]
        return PagedList(experiments, str(end) if end < len(self.experiment_ids) else None)


@pytest.fixture
def paged_client(monkeypatch):
    def install(experiment_ids: list[str]) -> _OffsetPagedClient:
        client = _OffsetPagedClient(experiment_ids)
        monkeypatch.setattr(mcp_tools, "MlflowClient", lambda: client)
        monkeypatch.setattr(mcp_tools, "get_mcp_request_username", lambda: "reader")
        monkeypatch.setattr(
            auth_module,
            "_role_based_read_predicate",
            lambda username, resource: lambda experiment_id: experiment_id.startswith("r"),
        )
        return client

    return install


def _walk(max_results: int | None) -> list[list[str]]:
    pages = []
    page_token = None
    while True:
        page = search_readable_experiments(max_results=max_results, page_token=page_token)
        pages.append([e.experiment_id for e in page.experiments])
        if (page_token := page.next_page_token) is None:
            return pages


@pytest.mark.parametrize("max_results", [1, 2, 3, 5, 100, None])
def test_search_readable_experiments_walks_interleaved_rows_without_gaps_or_duplicates(
    paged_client, max_results
):
    # Readable ("r") and unreadable ("u") rows interleaved irregularly, including runs of each.
    layout = "rurruuurrrruuuuuurururrr"
    experiment_ids = [f"{kind}{i}" for i, kind in enumerate(layout)]
    client = paged_client(experiment_ids)

    pages = _walk(max_results)

    readable = [e for e in experiment_ids if e.startswith("r")]
    assert [e for page in pages for e in page] == readable
    if max_results is not None:
        assert all(len(page) <= max_results for page in pages)
        # Every page except the last one is full.
        assert all(len(page) == max_results for page in pages[:-1])
        # Each store request asks for exactly the remaining slots of the page.
        assert all(0 < size <= max_results for size in client.page_sizes)


def test_search_readable_experiments_returns_the_token_of_a_full_page(paged_client):
    paged_client(["r0", "r1", "u2", "r3"])
    page = search_readable_experiments(max_results=2)
    assert [e.experiment_id for e in page.experiments] == ["r0", "r1"]
    assert page.next_page_token == "2"

    page = search_readable_experiments(max_results=2, page_token=page.next_page_token)
    assert [e.experiment_id for e in page.experiments] == ["r3"]
    assert page.next_page_token is None


def test_search_readable_experiments_stops_at_the_store_page_cap(paged_client):
    # Only the row after the capped pages is readable.
    cap = SEARCH_READABLE_EXPERIMENTS_MAX_STORE_PAGES
    client = paged_client([f"u{i}" for i in range(2 * cap)] + ["r-last"])

    first = search_readable_experiments(max_results=2)
    assert len(client.page_sizes) == cap
    assert first.experiments == []
    assert first.next_page_token == str(2 * cap)

    second = search_readable_experiments(max_results=2, page_token=first.next_page_token)
    assert [e.experiment_id for e in second.experiments] == ["r-last"]
    assert second.next_page_token is None


def test_search_readable_experiments_rejects_negative_max_results(paged_client):
    paged_client([])
    with pytest.raises(MlflowException, match="non-negative"):
        search_readable_experiments(max_results=-1)


@pytest.mark.asyncio
async def test_reader_can_search_experiments_with_no_grants_and_sees_nothing(
    mcp_server, monkeypatch
):
    _experiments(mcp_server, monkeypatch, ["exp-a"])
    nobody = create_user(mcp_server)
    assert await _call(mcp_server, nobody, "search_experiments") == {
        "experiments": [],
        "next_page_token": None,
    }


# --------------------------------------------------------------------------- creator grants


async def _assert_manages_experiment(url: str, credentials: tuple[str, str], exp_id: str) -> None:
    assert (await _call(url, credentials, "get_experiment", experiment_id=exp_id))[
        "experiment_id"
    ] == exp_id
    await _call(url, credentials, "rename_experiment", experiment_id=exp_id, new_name=f"r-{exp_id}")
    await _call(url, credentials, "delete_experiment", experiment_id=exp_id)


@pytest.mark.asyncio
async def test_creator_manages_the_experiment_it_creates(mcp_server):
    creator = create_user(mcp_server)
    created = await _call(mcp_server, creator, "create_experiment", experiment_name="mine")

    # Nobody else is granted anything on it.
    with pytest.raises(ToolError, match="^Permission denied$"):
        await _call(
            mcp_server,
            create_user(mcp_server),
            "get_experiment",
            experiment_id=created["experiment_id"],
        )
    await _assert_manages_experiment(mcp_server, creator, created["experiment_id"])


@pytest.mark.asyncio
async def test_create_run_grants_only_the_experiment_it_creates(mcp_server, monkeypatch):
    creator = create_user(mcp_server)
    run = await _call(mcp_server, creator, "create_run", experiment_name="made-by-run")
    await _assert_manages_experiment(mcp_server, creator, run["experiment_id"])

    # A run in an existing experiment, addressed by name, leaves the caller's grant as it was.
    (existing,) = _experiments(mcp_server, monkeypatch, ["existing"])
    editor = _reader(mcp_server, existing, permission="EDIT")
    run = await _call(mcp_server, editor, "create_run", experiment_name="existing")
    assert run["experiment_id"] == existing
    with pytest.raises(ToolError, match="^Permission denied$"):
        await _call(mcp_server, editor, "delete_experiment", experiment_id=existing)


@pytest.mark.asyncio
async def test_create_run_grants_the_experiment_it_creates_when_the_run_fails(mcp_server):
    creator = create_user(mcp_server)
    # The store rejects the run only after the tool has created the experiment.
    with pytest.raises(ToolError, match="Both 'run_name' argument and 'mlflow.runName' tag"):
        await _call(
            mcp_server,
            creator,
            "create_run",
            experiment_name="orphaned",
            run_name="b",
            tags={"mlflow.runName": "a"},
        )

    experiment = await _call(mcp_server, creator, "get_experiment", experiment_name="orphaned")
    assert experiment["name"] == "orphaned"
    await _call(mcp_server, creator, "delete_experiment", experiment_id=experiment["experiment_id"])


def _scorer_request(url: str, method: str, credentials, path: str, **params) -> requests.Response:
    body = {"params": params} if method == "GET" else {"json": params}
    return requests.request(method, f"{url}/api/3.0/mlflow/{path}", auth=credentials, **body)


@pytest.mark.asyncio
async def test_creator_manages_the_scorer_it_registers(mcp_server, monkeypatch):
    (exp_id,) = _experiments(mcp_server, monkeypatch, ["exp-a"])
    # EDIT on the experiment allows registering; it confers nothing on the experiment's scorers.
    creator = _reader(mcp_server, exp_id, permission="EDIT")
    other_username, _ = create_user(mcp_server)

    registered = await _call(
        mcp_server,
        creator,
        "register_llm_judge_scorer",
        name="judge",
        instructions="Is {{ outputs }} correct?",
        experiment_id=exp_id,
    )
    assert registered == {"name": "judge", "experiment_id": exp_id}

    get = _scorer_request(
        mcp_server, "GET", creator, "scorers/get", experiment_id=exp_id, name="judge"
    )
    assert get.status_code == 200
    listing = await _call(mcp_server, creator, "list_scorers", experiment_id=exp_id)
    assert [s["name"] for s in listing["scorers"]] == ["judge"]
    # No REST scorer route is gated on update; granting access to another user needs MANAGE.
    grant = _scorer_request(
        mcp_server,
        "POST",
        creator,
        "users/permissions/grant",
        username=other_username,
        resource_type="scorer",
        resource_id=f"{exp_id}/judge",
        permission="READ",
    )
    assert grant.status_code == 200
    delete = _scorer_request(
        mcp_server, "DELETE", creator, "scorers/delete", experiment_id=exp_id, name="judge"
    )
    assert delete.status_code == 200


@pytest.mark.asyncio
async def test_list_scorers_returns_only_the_scorers_the_caller_can_read(mcp_server, monkeypatch):
    (exp_id,) = _experiments(mcp_server, monkeypatch, ["exp-a"])
    for name in ("visible", "hidden"):
        await _call(
            mcp_server,
            ADMIN,
            "register_llm_judge_scorer",
            name=name,
            instructions="Is {{ outputs }} correct?",
            experiment_id=exp_id,
        )
    # Experiment READ passes the list gate but grants nothing on the scorers themselves.
    reader = _reader(mcp_server, exp_id)
    assert await _call(mcp_server, reader, "list_scorers", experiment_id=exp_id) == {"scorers": []}

    grant_role_permission(mcp_server, reader[0], "scorer", f"{exp_id}/visible", "READ")
    listing = await _call(mcp_server, reader, "list_scorers", experiment_id=exp_id)
    assert [s["name"] for s in listing["scorers"]] == ["visible"]
    listing = await _call(mcp_server, ADMIN, "list_scorers", experiment_id=exp_id)
    assert sorted(s["name"] for s in listing["scorers"]) == ["hidden", "visible"]
    # The built-in catalog is not filtered.
    builtin = await _call(mcp_server, reader, "list_scorers", builtin=True)
    assert builtin == await _call(mcp_server, ADMIN, "list_scorers", builtin=True)


@pytest.mark.parametrize("mcp_server", [{"MLFLOW_BASIC_AUTH_FAIL_CLOSED": "true"}], indirect=True)
@pytest.mark.asyncio
async def test_fail_closed_mode_keeps_the_authenticated_endpoint_reachable(mcp_server, monkeypatch):
    (exp_a,) = _experiments(mcp_server, monkeypatch, ["exp-a"])
    reader = _reader(mcp_server, exp_a)
    experiment = await _call(mcp_server, reader, "get_experiment", experiment_id=exp_a)
    assert experiment["name"] == "exp-a"


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
        assert result.structured_content["name"] == "exp-a"
        with pytest.raises(ToolError, match="^Permission denied$"):
            await client.call_tool("get_experiment", {"experiment_id": exp_b})


# --------------------------------------------------------------------------- workspaces


@pytest.fixture
def workspace_mcp_server(tmp_path):
    backend_uri = _backend_uri(tmp_path)
    extra_env = {
        MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key",
        "MLFLOW_AUTH_CONFIG_PATH": str(_write_auth_config(tmp_path)),
        "_MLFLOW_SGI_NAME": "uvicorn",
        MLFLOW_SERVER_ENABLE_MCP.name: "true",
        MLFLOW_ENABLE_WORKSPACES.name: "true",
        MLFLOW_WORKSPACE_STORE_URI.name: backend_uri,
        MLFLOW_RBAC_SEED_DEFAULT_ROLES.name: "true",
    }
    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        extra_env=extra_env,
        app="mlflow.server.auth:create_app",
        server_type="fastapi",
    ) as url:
        yield url


def _create_workspace(url: str, name: str) -> None:
    requests.post(
        f"{url}/api/3.0/mlflow/workspaces", json={"name": name}, auth=ADMIN
    ).raise_for_status()


def _create_experiment_in_workspace(url: str, workspace: str, name: str) -> str:
    response = requests.post(
        f"{url}/api/2.0/mlflow/experiments/create",
        json={"name": name},
        auth=ADMIN,
        headers={WORKSPACE_HEADER_NAME: workspace},
    )
    response.raise_for_status()
    return response.json()["experiment_id"]


@pytest.mark.asyncio
async def test_workspace_header_scopes_tool_results_and_permission_checks(workspace_mcp_server):
    url = workspace_mcp_server
    _create_workspace(url, "ws-a")
    _create_workspace(url, "ws-b")
    exp_a = _create_experiment_in_workspace(url, "ws-a", "exp-in-a")
    exp_b = _create_experiment_in_workspace(url, "ws-b", "exp-in-b")
    username, password = create_user(url)
    grant_role_permission(url, username, "experiment", exp_a, "READ", workspace="ws-a")
    reader = (username, password)

    # (a) The header selects the workspace the tools and the permission checks run in.
    experiment = await _call(url, reader, "get_experiment", workspace="ws-a", experiment_id=exp_a)
    assert experiment["name"] == "exp-in-a"
    listing = await _call(url, reader, "search_experiments", workspace="ws-a")
    assert _names(listing) == ["exp-in-a"]
    assert _names(await _call(url, reader, "search_experiments", workspace="ws-b")) == []

    # (b) The grant lives in workspace A: the same experiment id addressed through workspace B
    # is denied, as is an experiment that belongs to another workspace.
    with pytest.raises(ToolError, match="^Permission denied$"):
        await _call(url, reader, "get_experiment", workspace="ws-b", experiment_id=exp_a)
    with pytest.raises(ToolError, match="^Permission denied$"):
        await _call(url, reader, "get_experiment", workspace="ws-a", experiment_id=exp_b)


@pytest.mark.asyncio
async def test_creator_grant_lands_in_the_request_workspace(workspace_mcp_server):
    url = workspace_mcp_server
    _create_workspace(url, "ws-a")
    username, password = create_user(url)
    grant_role_permission(url, username, "workspace", "*", "USE", workspace="ws-a")
    creator = (username, password)

    created = await _call(
        url, creator, "create_experiment", workspace="ws-a", experiment_name="mine"
    )
    exp_id = created["experiment_id"]
    experiment = await _call(url, creator, "get_experiment", workspace="ws-a", experiment_id=exp_id)
    assert experiment["name"] == "mine"
    await _call(url, creator, "delete_experiment", workspace="ws-a", experiment_id=exp_id)


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
    served = {tool.name for tool in SHARED_TOOLS}
    assert served == set(MCP_TOOL_RULES)
    check_mcp_tool_coverage(served)


def test_policy_hooks_name_served_tools():
    policy = mcp_tools.get_mcp_tool_policy()
    served = {tool.name for tool in SHARED_TOOLS}
    assert set(policy.overrides) <= served
    assert set(policy.on_success) == {"create_experiment", "register_llm_judge_scorer"}


@pytest.mark.parametrize(
    ("override", "tool"),
    [(search_readable_experiments, search_experiments), (list_readable_scorers, list_scorers)],
)
def test_override_has_the_signature_of_the_tool(override, tool):
    assert list(inspect.signature(override).parameters) == list(inspect.signature(tool).parameters)


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
