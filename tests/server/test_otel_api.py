from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from mlflow.entities import Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.server.fastapi_app import add_fastapi_workspace_middleware
from mlflow.server.otel_api import otel_router
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH
from mlflow.tracking._workspace import context as workspace_context
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME


def _build_otlp_payload():
    request = ExportTraceServiceRequest()
    span = request.resource_spans.add().scope_spans.add().spans.add()
    span.trace_id = b"\x00" * 16
    span.span_id = b"\x01" * 8
    span.name = "span"
    return request.SerializeToString()


def _make_test_client():
    app = FastAPI()
    add_fastapi_workspace_middleware(app)
    app.include_router(otel_router)
    return TestClient(app)


def test_workspace_scoped_otlp_endpoint_sets_workspace(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    class DummyTrackingStore:
        def __init__(self):
            self.calls = []

        def log_spans(self, experiment_id, spans):
            self.calls.append((workspace_context.get_current_workspace(), experiment_id, spans))

    tracking_store = DummyTrackingStore()
    captured = {}

    def fake_resolve(header_workspace):
        captured["requested"] = header_workspace
        return Workspace(name=header_workspace)

    monkeypatch.setattr(
        "mlflow.server.fastapi_app.resolve_workspace_from_header",
        fake_resolve,
    )
    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: tracking_store,
    )

    client = _make_test_client()
    response = client.post(
        OTLP_TRACES_PATH,
        data=_build_otlp_payload(),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "42",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )

    assert response.status_code == 200
    assert captured["requested"].strip() == "team-a"
    assert tracking_store.calls[0][0] == "team-a"
    # Workspace context should be cleared after the request
    assert workspace_context.get_current_workspace() is None


def test_default_otlp_endpoint_uses_default_workspace(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    class DummyTrackingStore:
        def __init__(self):
            self.calls = []

        def log_spans(self, experiment_id, spans):
            self.calls.append((workspace_context.get_current_workspace(), experiment_id, spans))

    tracking_store = DummyTrackingStore()
    captured = {}

    def fake_resolve(header_workspace):
        captured["requested"] = header_workspace
        return Workspace(name="default")

    monkeypatch.setattr(
        "mlflow.server.fastapi_app.resolve_workspace_from_header",
        fake_resolve,
    )
    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: tracking_store,
    )

    client = _make_test_client()
    response = client.post(
        OTLP_TRACES_PATH,
        data=_build_otlp_payload(),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "7",
        },
    )

    assert response.status_code == 200
    assert captured["requested"] is None
    assert tracking_store.calls[0][0] == "default"
    assert workspace_context.get_current_workspace() is None


def test_otlp_endpoint_without_default_workspace_raises_error(monkeypatch):
    """Test that missing default workspace raises appropriate error."""
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    def fake_resolve(_header_workspace):
        return None

    monkeypatch.setattr(
        "mlflow.server.fastapi_app.resolve_workspace_from_header",
        fake_resolve,
    )

    client = _make_test_client()
    response = client.post(
        OTLP_TRACES_PATH,
        data=_build_otlp_payload(),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "42",
        },
    )

    assert response.status_code == 400
    assert "Active workspace is required" in response.json()["message"]


def test_otlp_invalid_content_type(monkeypatch):
    """Test that invalid Content-Type header returns HTTP 400."""
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")

    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: mock.Mock(),
    )

    client = _make_test_client()

    # Test with wrong content type
    response = client.post(
        OTLP_TRACES_PATH,
        data=_build_otlp_payload(),
        headers={
            "Content-Type": "application/json",
            "X-MLflow-Experiment-Id": "42",
        },
    )
    assert response.status_code == 400
    assert "Invalid Content-Type" in response.json()["detail"]

    # Test with missing content type
    response = client.post(
        OTLP_TRACES_PATH,
        data=_build_otlp_payload(),
        headers={
            "X-MLflow-Experiment-Id": "42",
        },
    )
    assert response.status_code == 400
    assert "Invalid Content-Type" in response.json()["detail"]


def test_otlp_invalid_protobuf_data(monkeypatch):
    """Test that invalid protobuf data returns HTTP 400."""
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")

    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: mock.Mock(),
    )

    client = _make_test_client()

    # Test with invalid protobuf data
    response = client.post(
        OTLP_TRACES_PATH,
        data=b"this is not valid protobuf data",
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "42",
        },
    )
    assert response.status_code == 400
    assert "Invalid OpenTelemetry protobuf format" in response.json()["detail"]


def test_otlp_empty_resource_spans(monkeypatch):
    """Test that empty resource spans returns HTTP 400."""
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")

    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: mock.Mock(),
    )

    client = _make_test_client()

    # Create request with no resource spans
    request = ExportTraceServiceRequest()

    response = client.post(
        OTLP_TRACES_PATH,
        data=request.SerializeToString(),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "42",
        },
    )
    assert response.status_code == 400
    assert "no spans found" in response.json()["detail"]


def test_otlp_conversion_error(monkeypatch):
    """Test that span conversion errors return HTTP 422."""
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")

    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: mock.Mock(),
    )

    # Mock Span.from_otel_proto to raise exception
    def mock_from_otel_proto(proto_span):
        raise Exception("Cannot convert span")

    monkeypatch.setattr(
        "mlflow.entities.span.Span.from_otel_proto",
        mock_from_otel_proto,
    )

    client = _make_test_client()

    response = client.post(
        OTLP_TRACES_PATH,
        data=_build_otlp_payload(),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "42",
        },
    )
    assert response.status_code == 422
    assert "Cannot convert OpenTelemetry span" in response.json()["detail"]
