from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS


def _make_export_request_bytes(*, service_names: list[str] | None) -> bytes:
    export_request = ExportTraceServiceRequest()

    service_names_to_emit = service_names if service_names is not None else [None]

    for service_name in service_names_to_emit:
        resource_span = export_request.resource_spans.add()
        if service_name is not None:
            attr = resource_span.resource.attributes.add()
            attr.key = "service.name"
            attr.value.string_value = service_name

        scope_span = resource_span.scope_spans.add()
        span = scope_span.spans.add()
        span.name = "test"
        span.trace_id = b"\x00" * 15 + b"\x01"
        span.span_id = b"\x00" * 7 + b"\x02"
        span.start_time_unix_nano = 1
        span.end_time_unix_nano = 2

    return export_request.SerializeToString()


@pytest.fixture
def otel_client(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ALLOWED_HOSTS", "*")

    from mlflow.server.fastapi_app import create_fastapi_app

    with TestClient(create_fastapi_app()) as client:
        yield client


def test_export_traces_uses_experiment_id_header(otel_client):
    store = MagicMock()

    with patch("mlflow.server.otel_api._get_tracking_store", return_value=store):
        response = otel_client.post(
            "/v1/traces",
            headers={
                "Content-Type": "application/x-protobuf",
                "x-mlflow-experiment-id": "123",
            },
            content=_make_export_request_bytes(service_names=["svc"]),
        )

    assert response.status_code == 200
    store.log_spans.assert_called_once()
    assert store.log_spans.call_args[0][0] == "123"
    store.get_experiment_by_name.assert_not_called()


def test_export_traces_falls_back_to_service_name_existing_experiment(otel_client):
    store = MagicMock()
    store.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-1")

    with patch("mlflow.server.otel_api._get_tracking_store", return_value=store):
        response = otel_client.post(
            "/v1/traces",
            headers={"Content-Type": "application/x-protobuf"},
            content=_make_export_request_bytes(service_names=["svc"]),
        )

    assert response.status_code == 200
    store.get_experiment_by_name.assert_called_once_with("svc")
    store.create_experiment.assert_not_called()
    store.log_spans.assert_called_once()
    assert store.log_spans.call_args[0][0] == "exp-1"


def test_export_traces_falls_back_to_service_name_creates_experiment(otel_client):
    store = MagicMock()
    store.get_experiment_by_name.return_value = None
    store.create_experiment.return_value = "exp-new"

    with patch("mlflow.server.otel_api._get_tracking_store", return_value=store):
        response = otel_client.post(
            "/v1/traces",
            headers={"Content-Type": "application/x-protobuf"},
            content=_make_export_request_bytes(service_names=["svc"]),
        )

    assert response.status_code == 200
    store.get_experiment_by_name.assert_called_once_with("svc")
    store.create_experiment.assert_called_once_with("svc")
    store.log_spans.assert_called_once()
    assert store.log_spans.call_args[0][0] == "exp-new"


def test_export_traces_requires_header_or_service_name(otel_client):
    store = MagicMock()

    with patch("mlflow.server.otel_api._get_tracking_store", return_value=store):
        response = otel_client.post(
            "/v1/traces",
            headers={"Content-Type": "application/x-protobuf"},
            content=_make_export_request_bytes(service_names=None),
        )

    assert response.status_code == 400
    assert "x-mlflow-experiment-id" in response.json()["detail"]
    store.log_spans.assert_not_called()


def test_export_traces_rejects_multiple_service_names(otel_client):
    store = MagicMock()

    with patch("mlflow.server.otel_api._get_tracking_store", return_value=store):
        response = otel_client.post(
            "/v1/traces",
            headers={"Content-Type": "application/x-protobuf"},
            content=_make_export_request_bytes(service_names=["svc-a", "svc-b"]),
        )

    assert response.status_code == 400
    assert "multiple" in response.json()["detail"]
    store.log_spans.assert_not_called()


def test_export_traces_requires_header_when_service_name_is_blank(otel_client):
    store = MagicMock()

    with patch("mlflow.server.otel_api._get_tracking_store", return_value=store):
        response = otel_client.post(
            "/v1/traces",
            headers={"Content-Type": "application/x-protobuf"},
            content=_make_export_request_bytes(service_names=["   "]),
        )

    assert response.status_code == 400
    assert "x-mlflow-experiment-id" in response.json()["detail"]
    store.log_spans.assert_not_called()


def test_export_traces_handles_experiment_creation_race(otel_client):
    store = MagicMock()
    store.get_experiment_by_name.side_effect = [None, MagicMock(experiment_id="exp-raced")]
    store.create_experiment.side_effect = MlflowException(
        "exists", error_code=RESOURCE_ALREADY_EXISTS
    )

    with patch("mlflow.server.otel_api._get_tracking_store", return_value=store):
        response = otel_client.post(
            "/v1/traces",
            headers={"Content-Type": "application/x-protobuf"},
            content=_make_export_request_bytes(service_names=["svc"]),
        )

    assert response.status_code == 200
    assert store.get_experiment_by_name.call_count == 2
    store.log_spans.assert_called_once()
    assert store.log_spans.call_args[0][0] == "exp-raced"
