from pathlib import Path

import pytest
import requests

import mlflow
from mlflow.server import handlers
from mlflow.server.fastapi_app import app
from mlflow.server.handlers import initialize_backend_stores

from tests.helper_functions import get_safe_port
from tests.tracking.integration_test_utils import ServerThread


@pytest.fixture
def tracking_server(tmp_path: Path):
    backend_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    handlers._tracking_store = None
    handlers._model_registry_store = None
    initialize_backend_stores(backend_uri, default_artifact_root=tmp_path.as_uri())

    with ServerThread(app, get_safe_port()) as url:
        mlflow.set_tracking_uri(url)
        yield url
        mlflow.set_tracking_uri(None)


def test_generate_demo_route_creates_data(tracking_server):
    response = requests.post(f"{tracking_server}/ajax-api/3.0/mlflow/demo/generate")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "created"
    assert data["experiment_id"] is not None
    assert len(data["features_generated"]) > 0
    assert data["navigation_url"] is not None


def test_generate_demo_route_idempotent(tracking_server):
    response1 = requests.post(f"{tracking_server}/ajax-api/3.0/mlflow/demo/generate")
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["status"] == "created"

    response2 = requests.post(f"{tracking_server}/ajax-api/3.0/mlflow/demo/generate")
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["status"] == "exists"
    assert data2["features_generated"] == []


def test_delete_demo_route_removes_data(tracking_server):
    requests.post(f"{tracking_server}/ajax-api/3.0/mlflow/demo/generate")

    response = requests.post(f"{tracking_server}/ajax-api/3.0/mlflow/demo/delete")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    assert len(data["features_deleted"]) > 0


def test_delete_demo_route_when_no_data(tracking_server):
    response = requests.post(f"{tracking_server}/ajax-api/3.0/mlflow/demo/delete")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    assert data["features_deleted"] == []


def test_generate_after_delete_regenerates(tracking_server):
    response1 = requests.post(f"{tracking_server}/ajax-api/3.0/mlflow/demo/generate")
    assert response1.json()["status"] == "created"

    requests.post(f"{tracking_server}/ajax-api/3.0/mlflow/demo/delete")

    response2 = requests.post(f"{tracking_server}/ajax-api/3.0/mlflow/demo/generate")
    assert response2.status_code == 200
    data = response2.json()
    assert data["status"] == "created"
    assert len(data["features_generated"]) > 0


def test_get_demo_endpoints_returns_routes():
    from mlflow.server.handlers import get_demo_endpoints

    endpoints = get_demo_endpoints()

    assert len(endpoints) == 2

    paths = [path for path, _, _ in endpoints]
    assert any("/mlflow/demo/generate" in path for path in paths)
    assert any("/mlflow/demo/delete" in path for path in paths)


def test_get_endpoints_includes_demo():
    from mlflow.server.handlers import get_endpoints

    endpoints = get_endpoints()
    paths = [path for path, _, _ in endpoints]

    assert any("/mlflow/demo/generate" in path for path in paths)
    assert any("/mlflow/demo/delete" in path for path in paths)
