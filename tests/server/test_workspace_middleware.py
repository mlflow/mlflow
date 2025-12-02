from __future__ import annotations

import pytest
import werkzeug
from fastapi import FastAPI
from fastapi.testclient import TestClient
from flask import Flask

from mlflow.entities import Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.server import app as flask_app
from mlflow.server.fastapi_app import add_fastapi_workspace_middleware
from mlflow.server.job_api import job_api_router
from mlflow.server.workspace_helpers import (
    WORKSPACE_HEADER_NAME,
    workspace_before_request_handler,
    workspace_teardown_request_handler,
)
from mlflow.utils import workspace_context


@pytest.fixture
def flask_workspace_app(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    if not hasattr(werkzeug, "__version__"):
        werkzeug.__version__ = "tests"

    app = Flask(__name__)
    app.before_request(workspace_before_request_handler)
    app.teardown_request(workspace_teardown_request_handler)

    @app.route("/ping")
    def _ping():
        return workspace_context.get_request_workspace() or "none"

    return app


def test_flask_workspace_middleware_sets_context(flask_workspace_app, monkeypatch):
    class DummyWorkspaceStore:
        def get_workspace(self, name):
            return Workspace(name=name)

    store = DummyWorkspaceStore()
    monkeypatch.setattr(
        "mlflow.server.workspace_helpers._get_workspace_store",
        lambda workspace_uri=None, tracking_uri=None: store,
    )

    client = flask_workspace_app.test_client()
    resp = client.get("/ping", headers={WORKSPACE_HEADER_NAME: "team-a"})
    assert resp.data.decode() == "team-a"
    assert workspace_context.get_request_workspace() is None


def test_flask_workspace_middleware_requires_header(flask_workspace_app, monkeypatch):
    class DefaultlessWorkspaceStore:
        def get_default_workspace(self):
            raise MlflowException.invalid_parameter_value("Active workspace is required.")

    store = DefaultlessWorkspaceStore()
    monkeypatch.setattr(
        "mlflow.server.workspace_helpers._get_workspace_store",
        lambda workspace_uri=None, tracking_uri=None: store,
    )

    client = flask_workspace_app.test_client()
    resp = client.get("/ping")
    assert resp.status_code == 400
    assert "Active workspace is required" in resp.json["message"]
    assert workspace_context.get_request_workspace() is None


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
        "mlflow.server.fastapi_app.resolve_workspace_from_header",
        lambda header: Workspace(name=header),
    )

    client = TestClient(app)
    resp = client.get(ping_path, headers={WORKSPACE_HEADER_NAME: "team-fast"})
    assert resp.status_code == 200
    assert resp.json() == {"workspace": "team-fast"}
    assert workspace_context.get_request_workspace() is None


def test_fastapi_workspace_middleware_requires_header(monkeypatch):
    app, ping_path = _fastapi_workspace_app(monkeypatch)
    monkeypatch.setattr(
        "mlflow.server.fastapi_app.resolve_workspace_from_header",
        lambda header: None,
    )

    client = TestClient(app)
    resp = client.get(ping_path)
    assert resp.status_code == 400
    assert "Active workspace is required" in resp.json()["message"]
    assert workspace_context.get_request_workspace() is None


def test_server_features_endpoint(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    client = flask_app.test_client()
    resp = client.get("/api/2.0/mlflow/server-features")
    assert resp.status_code == 200
    assert resp.get_json() == {"workspaces_enabled": True}

    # Disable workspaces and ensure the endpoint reflects the change.
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    resp = client.get("/api/2.0/mlflow/server-features")
    assert resp.status_code == 200
    assert resp.get_json() == {"workspaces_enabled": False}
