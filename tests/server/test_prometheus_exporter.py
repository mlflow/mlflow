import os
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

from mlflow.server.prometheus_exporter import activate_prometheus_exporter

tmpdir = TemporaryDirectory()


@pytest.fixture()
def app():
    from mlflow.server import app

    ctx = app.app_context()
    ctx.push()
    yield app
    ctx.pop()


@mock.patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": tmpdir.name})
def test_metrics(app):
    metrics = activate_prometheus_exporter(app)
    metric_labels = {"method": "GET", "status": "200"}

    with app.test_client() as c:
        assert (
            metrics.registry.get_sample_value("mlflow_http_request_total", labels=metric_labels)
            is None
        )
        c.get("/")
        assert (
            metrics.registry.get_sample_value("mlflow_http_request_total", labels=metric_labels)
            == 1
        )

        # calling the metrics endpoint should not increment the counter
        c.get("/metrics")
        assert (
            metrics.registry.get_sample_value("mlflow_http_request_total", labels=metric_labels)
            == 1
        )

        # calling the health endpoint should not increment the counter
        c.get("/health")
        assert (
            metrics.registry.get_sample_value("mlflow_http_request_total", labels=metric_labels)
            == 1
        )
