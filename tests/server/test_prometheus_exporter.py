import os
from unittest import mock

import pytest


from mlflow.server.prometheus_exporter import activate_prometheus_exporter


@pytest.fixture(autouse=True)
def mock_settings_env_vars(tmpdir):
    with mock.patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": tmpdir.strpath}):
        yield


@pytest.fixture()
def app():
    from mlflow.server import app

    with app.app_context():
        yield app


@pytest.fixture()
def test_client(app):
    with app.test_client() as c:
        yield c


def test_metrics(app, test_client):
    metrics = activate_prometheus_exporter(app)

    # test metrics for successful responses
    success_labels = {"method": "GET", "status": "200"}
    assert (
        metrics.registry.get_sample_value("mlflow_http_request_total", labels=success_labels)
        is None
    )
    resp = test_client.get("/")
    assert resp.status_code == 200
    assert (
        metrics.registry.get_sample_value("mlflow_http_request_total", labels=success_labels) == 1
    )

    # calling the metrics endpoint should not increment the counter
    resp = test_client.get("/metrics")
    assert resp.status_code == 200
    assert (
        metrics.registry.get_sample_value("mlflow_http_request_total", labels=success_labels) == 1
    )

    # calling the health endpoint should not increment the counter
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert (
        metrics.registry.get_sample_value("mlflow_http_request_total", labels=success_labels) == 1
    )

    # test metrics for failed responses
    failure_labels = {"method": "GET", "status": "404"}
    assert (
        metrics.registry.get_sample_value("mlflow_http_request_total", labels=failure_labels)
        is None
    )
    resp = test_client.get("/non-existent-endpoint")
    assert resp.status_code == 404
    assert (
        metrics.registry.get_sample_value("mlflow_http_request_total", labels=failure_labels) == 1
    )
