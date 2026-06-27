from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlflow.entities import Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.server.fastapi_app import add_fastapi_workspace_middleware, create_fastapi_app
from mlflow.server.job_api import job_api_router
from mlflow.server.workspace_helpers import (
    WORKSPACE_HEADER_NAME,
    resolve_workspace_for_request_if_enabled,
)
from mlflow.utils import workspace_context


def _fastapi_workspace_app(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    app = FastAPI()
    add_fastapi_workspace_middleware(app)

    ping_path = f"{job_api_router.prefix}/ping"

    @app.get(ping_path)
    async def ping():
        return {"workspace": workspace_context.get_request_workspace()}

    return app, ping_path


def test_fastapi_workspace_middleware_sets_context(monkeypatch):
    app, ping_path = _fastapi_workspace_app(monkeypatch)
    monkeypatch.setattr(
        "mlflow.server.fastapi_app.resolve_workspace_for_request_if_enabled",
        lambda _path, header: Workspace(name=header),
    )

    client = TestClient(app)
    resp = client.get(ping_path, headers={WORKSPACE_HEADER_NAME: "team-fast"})
    assert resp.status_code == 200
    assert resp.json() == {"workspace": "team-fast"}
    assert workspace_context.get_request_workspace() is None


def test_fastapi_workspace_middleware_handles_missing_header(monkeypatch):
    app, ping_path = _fastapi_workspace_app(monkeypatch)
    monkeypatch.setattr(
        "mlflow.server.fastapi_app.resolve_workspace_for_request_if_enabled",
        lambda _path, _header: None,
    )

    client = TestClient(app)
    resp = client.get(ping_path)
    assert resp.status_code == 200
    assert resp.json() == {"workspace": None}
    assert workspace_context.get_request_workspace() is None


def test_fastapi_workspace_middleware_prefers_routed_path(monkeypatch):
    app, ping_path = _fastapi_workspace_app(monkeypatch)
    seen: dict[str, str] = {}

    def resolve(path, header):
        seen["path"] = path
        return Workspace(name=header)

    monkeypatch.setattr(
        "mlflow.server.fastapi_app.resolve_workspace_for_request_if_enabled",
        resolve,
    )

    client = TestClient(app)
    resp = client.get(
        ping_path,
        headers={
            WORKSPACE_HEADER_NAME: "team-fast",
            "Host": "example.com/api/3.0/mlflow/server-info?x=",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"workspace": "team-fast"}
    assert seen["path"] == ping_path
    assert workspace_context.get_request_workspace() is None


def test_server_info_workspaces_enabled(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    app = create_fastapi_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/3.0/mlflow/server-info", headers={"Host": "localhost"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["workspaces_enabled"] is True
    assert data["trace_archival_enabled"] is False

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    resp = client.get("/api/3.0/mlflow/server-info", headers={"Host": "localhost"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["workspaces_enabled"] is False
    assert data["trace_archival_enabled"] is False


def test_server_info_skips_workspace_resolution(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    resolved = {}

    _real = resolve_workspace_for_request_if_enabled

    def _spy(path, header_workspace):
        result = _real(path, header_workspace)
        resolved["path"] = path
        resolved["result"] = result
        return result

    monkeypatch.setattr(
        "mlflow.server.fastapi_app.resolve_workspace_for_request_if_enabled", _spy
    )

    app = create_fastapi_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get(
        "/api/3.0/mlflow/server-info",
        headers={"Host": "localhost", WORKSPACE_HEADER_NAME: "missing"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["workspaces_enabled"] is True
    assert data["trace_archival_enabled"] is False
    assert resolved["result"] is None


def test_server_info_with_workspace_header_when_workspaces_disabled(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    app = create_fastapi_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get(
        "/api/3.0/mlflow/server-info",
        headers={"Host": "localhost", WORKSPACE_HEADER_NAME: "some-workspace"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["workspaces_enabled"] is False
    assert data["trace_archival_enabled"] is False
