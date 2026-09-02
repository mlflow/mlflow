import asyncio
import threading
from abc import ABC
from unittest import mock
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from mlflow.entities import Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.server.fastapi_app import add_fastapi_workspace_middleware
from mlflow.server.otel_api import otel_router
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH, _set_otel_proto_anyvalue
from mlflow.utils import workspace_context
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME


def _build_otlp_payload(resource_attrs=None):
    request = ExportTraceServiceRequest()
    resource_span = request.resource_spans.add()
    if resource_attrs:
        for key, value in resource_attrs.items():
            attr = resource_span.resource.attributes.add()
            attr.key = key
            _set_otel_proto_anyvalue(attr.value, value)
    span = resource_span.scope_spans.add().spans.add()
    span.trace_id = b"\x00" * 16
    span.span_id = b"\x01" * 8
    span.name = "span"
    return request.SerializeToString()


def _make_test_client():
    app = FastAPI()
    add_fastapi_workspace_middleware(app)
    app.include_router(otel_router)
    return TestClient(app)


class _DummyTrackingStore:
    """Test store that offloads ``log_spans`` the same way production stores do."""

    async def log_spans_async(self, location, spans):
        return await asyncio.to_thread(self.log_spans, location, spans)


def test_workspace_scoped_otlp_endpoint_sets_workspace(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    class DummyTrackingStore(_DummyTrackingStore):
        def __init__(self):
            self.calls = []

        def log_spans(self, experiment_id, spans):
            self.calls.append((workspace_context.get_request_workspace(), experiment_id, spans))

    tracking_store = DummyTrackingStore()
    captured = {}

    def fake_resolve(_path, header_workspace):
        captured["requested"] = header_workspace
        return Workspace(name=header_workspace)

    monkeypatch.setattr(
        "mlflow.server.fastapi_app.resolve_workspace_for_request_if_enabled",
        fake_resolve,
    )
    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: tracking_store,
    )

    client = _make_test_client()
    response = client.post(
        OTLP_TRACES_PATH,
        content=_build_otlp_payload(),
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
    assert workspace_context.get_request_workspace() is None


def test_default_otlp_endpoint_uses_default_workspace(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    class DummyTrackingStore(_DummyTrackingStore):
        def __init__(self):
            self.calls = []

        def log_spans(self, experiment_id, spans):
            self.calls.append((workspace_context.get_request_workspace(), experiment_id, spans))

    tracking_store = DummyTrackingStore()
    captured = {}

    def fake_resolve(_path, header_workspace):
        captured["requested"] = header_workspace
        return Workspace(name="default")

    monkeypatch.setattr(
        "mlflow.server.fastapi_app.resolve_workspace_for_request_if_enabled",
        fake_resolve,
    )
    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: tracking_store,
    )

    client = _make_test_client()
    response = client.post(
        OTLP_TRACES_PATH,
        content=_build_otlp_payload(),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "7",
        },
    )

    assert response.status_code == 200
    assert captured["requested"] is None
    assert tracking_store.calls[0][0] == "default"
    assert workspace_context.get_request_workspace() is None


def test_otlp_endpoint_links_trace_to_run(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")

    class DummyTrackingStore(_DummyTrackingStore):
        def __init__(self):
            self.calls = []
            self.link_calls = []

        def log_spans(self, experiment_id, spans):
            self.calls.append((experiment_id, spans))

        def link_traces_to_run(self, trace_ids, run_id):
            self.link_calls.append((trace_ids, run_id))

    tracking_store = DummyTrackingStore()
    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: tracking_store,
    )

    client = _make_test_client()
    response = client.post(
        OTLP_TRACES_PATH,
        content=_build_otlp_payload(),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "42",
            "X-MLflow-Run-Id": "run-123",
        },
    )

    assert response.status_code == 200
    assert len(tracking_store.calls) == 1

    experiment_id, spans = tracking_store.calls[0]
    assert experiment_id == "42"
    assert len(spans) == 1
    assert spans[0].parent_id is None

    assert len(tracking_store.link_calls) == 1
    trace_ids, run_id = tracking_store.link_calls[0]
    assert trace_ids == [spans[0].trace_id]
    assert run_id == "run-123"


def test_otlp_endpoint_without_default_workspace_raises_error(monkeypatch):
    from mlflow.store.workspace_aware_mixin import WorkspaceAwareMixin

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    class DummyWorkspaceAwareStore(_DummyTrackingStore, WorkspaceAwareMixin):
        """A dummy store that raises MlflowException when workspace is not set."""

        def log_spans(self, experiment_id, spans):
            # This will raise MlflowException if workspace context is not set
            self._get_active_workspace()

    def fake_resolve(_path, _header_workspace):
        return None

    monkeypatch.setattr(
        "mlflow.server.fastapi_app.resolve_workspace_for_request_if_enabled",
        fake_resolve,
    )
    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: DummyWorkspaceAwareStore(),
    )

    client = _make_test_client()
    response = client.post(
        OTLP_TRACES_PATH,
        content=_build_otlp_payload(),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "42",
        },
    )

    assert response.status_code == 400
    assert "Active workspace is required" in response.json()["message"]


def test_otlp_endpoint_run_linking_error_is_logged(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    logged_messages = []

    class DummyTrackingStore(_DummyTrackingStore):
        def log_spans(self, experiment_id, spans):
            pass

        def link_traces_to_run(self, trace_ids, run_id):
            raise Exception("linking failed")

    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: DummyTrackingStore(),
    )
    monkeypatch.setattr(
        "mlflow.server.otel_api._logger.exception",
        lambda message: logged_messages.append(message),
    )

    client = _make_test_client()
    response = client.post(
        OTLP_TRACES_PATH,
        content=_build_otlp_payload(),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "42",
            "X-MLflow-Run-Id": "run-123",
        },
    )

    assert response.status_code == 200
    assert logged_messages == ["Failed to link OpenTelemetry traces to MLflow run"]


def test_otlp_invalid_content_type(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")

    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: mock.Mock(),
    )

    client = _make_test_client()

    # Test with unsupported content type
    response = client.post(
        OTLP_TRACES_PATH,
        content=_build_otlp_payload(),
        headers={
            "Content-Type": "text/plain",
            "X-MLflow-Experiment-Id": "42",
        },
    )
    assert response.status_code == 400
    assert "Invalid Content-Type" in response.json()["detail"]

    # Test with missing content type
    response = client.post(
        OTLP_TRACES_PATH,
        content=_build_otlp_payload(),
        headers={
            "X-MLflow-Experiment-Id": "42",
        },
    )
    assert response.status_code == 400
    assert "Invalid Content-Type" in response.json()["detail"]


def test_otlp_invalid_protobuf_data(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")

    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: mock.Mock(),
    )

    client = _make_test_client()

    # Test with invalid protobuf data
    response = client.post(
        OTLP_TRACES_PATH,
        content=b"this is not valid protobuf data",
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "42",
        },
    )
    assert response.status_code == 400
    assert "Invalid OpenTelemetry format" in response.json()["detail"]


def test_otlp_empty_resource_spans(monkeypatch):
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
        content=request.SerializeToString(),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "42",
        },
    )
    assert response.status_code == 400
    assert "no spans found" in response.json()["detail"]


def test_otlp_conversion_error(monkeypatch):
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
        content=_build_otlp_payload(),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "42",
        },
    )
    assert response.status_code == 422
    assert "Cannot convert OpenTelemetry span" in response.json()["detail"]


def test_otlp_resource_attributes_preserved(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")

    class DummyTrackingStore(_DummyTrackingStore):
        def __init__(self):
            self.logged_spans = []

        def log_spans(self, experiment_id, spans):
            self.logged_spans.extend(spans)

    tracking_store = DummyTrackingStore()
    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: tracking_store,
    )

    client = _make_test_client()
    resource_attrs = {
        "service.name": "my-service",
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
    }
    response = client.post(
        OTLP_TRACES_PATH,
        content=_build_otlp_payload(resource_attrs=resource_attrs),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-MLflow-Experiment-Id": "42",
        },
    )

    assert response.status_code == 200
    assert len(tracking_store.logged_spans) == 1
    span = tracking_store.logged_spans[0]
    res = dict(span._span.resource.attributes)
    assert res["service.name"] == "my-service"
    assert res["telemetry.sdk.language"] == "python"
    assert res["telemetry.sdk.name"] == "opentelemetry"


def _make_asgi_app():
    app = FastAPI()
    add_fastapi_workspace_middleware(app)
    app.include_router(otel_router)
    return app


async def _asgi_post(app, path: str, headers: dict[str, str], body: bytes) -> int:
    status_code = 500

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()],
            "client": ("testclient", 500),
            "server": ("test", 80),
        },
        receive,
        send,
    )
    return status_code


@pytest.mark.asyncio
async def test_otlp_export_offloads_store_io_from_event_loop(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")

    event_loop_ident = threading.get_ident()
    store_idents = []
    started = threading.Event()
    release = threading.Event()

    class DummyTrackingStore(_DummyTrackingStore):
        def log_spans(self, experiment_id, spans):
            store_idents.append(threading.get_ident())
            started.set()
            assert release.wait(timeout=5)

        def link_traces_to_run(self, trace_ids, run_id):
            store_idents.append(threading.get_ident())

    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: DummyTrackingStore(),
    )

    request_task = asyncio.create_task(
        _asgi_post(
            _make_asgi_app(),
            OTLP_TRACES_PATH,
            {
                "Content-Type": "application/x-protobuf",
                "X-MLflow-Experiment-Id": "42",
                "X-MLflow-Run-Id": "run-123",
            },
            _build_otlp_payload(),
        )
    )

    async def _wait_for_store_call():
        while not started.is_set():
            await asyncio.sleep(0.01)

    # If log_spans_async ran log_spans on the event loop, this wait would time
    # out because the loop would be blocked inside DummyTrackingStore.log_spans.
    await asyncio.wait_for(_wait_for_store_call(), timeout=5)
    await asyncio.sleep(0.01)
    release.set()
    status_code = await request_task

    assert status_code == 200
    assert store_idents
    assert all(ident != event_loop_ident for ident in store_idents)


@pytest.mark.asyncio
async def test_otlp_export_handles_concurrent_requests(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")

    barrier = threading.Barrier(2, timeout=5)

    class DummyTrackingStore(_DummyTrackingStore):
        def log_spans(self, experiment_id, spans):
            barrier.wait()

    monkeypatch.setattr(
        "mlflow.server.otel_api._get_tracking_store",
        lambda: DummyTrackingStore(),
    )

    app = _make_asgi_app()
    headers = {
        "Content-Type": "application/x-protobuf",
        "X-MLflow-Experiment-Id": "42",
    }
    payload = _build_otlp_payload()
    status_codes = await asyncio.gather(
        _asgi_post(app, OTLP_TRACES_PATH, headers, payload),
        _asgi_post(app, OTLP_TRACES_PATH, headers, payload),
    )

    assert status_codes == [200, 200]


@pytest.mark.asyncio
async def test_otlp_export_works_for_store_that_only_implements_log_spans(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")

    class SyncOnlyTrackingStore(AbstractStore, ABC):
        def log_spans(self, experiment_id, spans, tracking_uri=None):
            return spans

    with patch.multiple(SyncOnlyTrackingStore, __abstractmethods__=set()):
        monkeypatch.setattr(
            "mlflow.server.otel_api._get_tracking_store",
            lambda: SyncOnlyTrackingStore(),
        )
        status_code = await _asgi_post(
            _make_asgi_app(),
            OTLP_TRACES_PATH,
            {
                "Content-Type": "application/x-protobuf",
                "X-MLflow-Experiment-Id": "42",
            },
            _build_otlp_payload(),
        )

    assert status_code == 200
