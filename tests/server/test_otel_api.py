import asyncio
import base64
import gzip
import json
import threading
from abc import ABC
from unittest import mock
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span as OTelProtoSpan
from prometheus_client import REGISTRY

from mlflow.entities import Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.server.fastapi_app import add_fastapi_workspace_middleware, create_fastapi_app
from mlflow.server.otel_api import otel_router
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
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


@pytest.mark.parametrize(
    ("key", "sent_value", "expected"),
    [
        # MLflow's OTel exporter sends values already JSON-encoded, so a str output stays a str
        ("mlflow.spanOutputs", json.dumps('{"a": 1}'), '{"a": 1}'),
        # Raw JSON strings from third-party clients are parsed into structured values
        ("custom.attr", '{"a": 1}', {"a": 1}),
        ("custom.attr", "[1, 2]", [1, 2]),
        ("custom.attr", "hello", "hello"),
    ],
)
def test_otlp_json_string_attributes(monkeypatch, db_uri, tmp_path, key, sent_value, expected):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    store = SqlAlchemyStore(db_uri, tmp_path.as_uri())
    experiment_id = store.create_experiment("otlp-json-string-attributes")
    monkeypatch.setattr("mlflow.server.otel_api._get_tracking_store", lambda: store)

    trace_id = "0af7651916cd43dd8448eb211c80319c"
    span = {
        "traceId": trace_id,
        "spanId": "b7ad6b7169203331",
        "name": "span",
        "startTimeUnixNano": "1000000000",
        "endTimeUnixNano": "2000000000",
        "attributes": [{"key": key, "value": {"stringValue": sent_value}}],
    }
    response = _make_test_client().post(
        OTLP_TRACES_PATH,
        content=json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}),
        headers={"Content-Type": "application/json", "X-MLflow-Experiment-Id": experiment_id},
    )

    assert response.status_code == 200
    stored_span = store.get_trace(f"tr-{trace_id}").data.spans[0]
    assert stored_span.get_attribute(key) == expected


@pytest.mark.parametrize("content_type", ["application/json", "application/json; charset=utf-8"])
@pytest.mark.parametrize("compressed", [False, True])
def test_otlp_json_span_links_preserved(monkeypatch, db_uri, tmp_path, content_type, compressed):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    store = SqlAlchemyStore(db_uri, tmp_path.as_uri())
    experiment_id = store.create_experiment("otlp-json-links")
    monkeypatch.setattr("mlflow.server.otel_api._get_tracking_store", lambda: store)
    monkeypatch.setattr("mlflow.server.handlers._tracking_store", store)
    client = TestClient(create_fastapi_app(), base_url="http://localhost")

    trace_id = "0af7651916cd43dd8448eb211c80319c"
    span_id = "b7ad6b7169203331"
    linked_trace_id = "1234567890abcdef1234567890abcdef"
    linked_span_id = "1234567890abcdef"
    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": "span-with-links",
        "startTimeUnixNano": "1000000000",
        "endTimeUnixNano": "2000000000",
        "links": [
            {
                "traceId": linked_trace_id,
                "spanId": linked_span_id,
                "attributes": [{"key": "relationship", "value": {"stringValue": "follows"}}],
            },
            {"traceId": trace_id.upper(), "spanId": "ABCDEF0123456789"},
        ],
    }
    content = json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}).encode()
    headers = {"Content-Type": content_type, "X-MLflow-Experiment-Id": experiment_id}
    if compressed:
        content = gzip.compress(content)
        headers["Content-Encoding"] = "gzip"

    response = client.post(OTLP_TRACES_PATH, content=content, headers=headers)

    assert response.status_code == 200
    trace = store.get_trace(f"tr-{trace_id}")
    stored_span = trace.data.spans[0]
    assert stored_span.trace_id == f"tr-{trace_id}"
    assert stored_span.span_id == span_id
    assert [link.to_dict() for link in stored_span.links] == [
        {
            "trace_id": f"tr-{linked_trace_id}",
            "span_id": linked_span_id,
            "attributes": {"relationship": "follows"},
        },
        {"trace_id": f"tr-{trace_id}", "span_id": "abcdef0123456789", "attributes": None},
    ]
    response = client.get("/api/3.0/mlflow/traces/batchGet", params={"trace_ids": f"tr-{trace_id}"})
    assert response.status_code == 200
    response_links = response.json()["traces"][0]["spans"][0]["links"]
    assert [
        (base64.b64decode(link["trace_id"]).hex(), base64.b64decode(link["span_id"]).hex())
        for link in response_links
    ] == [
        (linked_trace_id, linked_span_id),
        (trace_id, "abcdef0123456789"),
    ]


@pytest.mark.parametrize("field", ["traceId", "spanId"])
def test_otlp_json_invalid_link_id(monkeypatch, field):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    span = {
        "traceId": "0af7651916cd43dd8448eb211c80319c",
        "spanId": "b7ad6b7169203331",
        "name": "invalid-link",
        "links": [{"traceId": "1234567890abcdef1234567890abcdef", "spanId": "1234567890abcdef"}],
    }
    span["links"][0][field] = "not-hex"
    response = _make_test_client().post(
        OTLP_TRACES_PATH,
        json={"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]},
        headers={"X-MLflow-Experiment-Id": "1"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid OpenTelemetry format"


@pytest.mark.parametrize("links", [None, []])
def test_otlp_json_without_links(monkeypatch, links):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    monkeypatch.setattr("mlflow.server.otel_api._get_tracking_store", lambda: _make_store()())
    span = {"traceId": "0" * 32, "spanId": "01" * 8, "name": "span", "links": links}
    response = _make_test_client().post(
        OTLP_TRACES_PATH,
        json={"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]},
        headers={"X-MLflow-Experiment-Id": "1"},
    )

    assert response.status_code == 200


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


def _build_multi_span_otlp_payload(span_count=3):
    request = ExportTraceServiceRequest()
    resource_span = request.resource_spans.add()
    scope_span = resource_span.scope_spans.add()

    trace_id = b"\x00" * 15 + b"\x01"
    root_span = scope_span.spans.add()
    root_span.trace_id = trace_id
    root_span.span_id = b"\x01" * 8
    root_span.name = "span-root"

    for i in range(1, span_count):
        child = scope_span.spans.add()
        child.trace_id = trace_id
        child.span_id = b"\x02" * 7 + bytes([i])
        child.parent_span_id = root_span.span_id
        child.name = f"span-{i}"

    return request.SerializeToString()


def _metric(name, labels=None):
    return REGISTRY.get_sample_value(name, labels) or 0.0


_OTLP_LABELS = {"protocol": "otlp"}
_PROTOBUF_HEADERS = {"Content-Type": "application/x-protobuf", "X-MLflow-Experiment-Id": "1"}


def _make_store(side_effect=None):
    class _Store(_DummyTrackingStore):
        def log_spans(self, experiment_id, spans):
            if side_effect is not None:
                raise side_effect

    return _Store


def test_otlp_prometheus_metrics(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")

    def _set_store(side_effect=None):
        monkeypatch.setattr(
            "mlflow.server.otel_api._get_tracking_store",
            lambda: _make_store(side_effect)(),
        )

    client = _make_test_client()

    def snap():
        return {
            "spans": _metric("mlflow_spans_ingested_total", _OTLP_LABELS),
            "errors": _metric("mlflow_trace_ingestion_server_errors_total", _OTLP_LABELS),
            "duration_count": _metric("mlflow_otlp_request_duration_seconds_count"),
            "payload_sum": _metric("mlflow_otlp_request_payload_bytes_sum"),
            "spans_per_req_sum": _metric("mlflow_otlp_spans_per_request_sum"),
        }

    def delta(before, after, key):
        return after[key] - before[key]

    # -- successful 5-span ingestion --
    _set_store()
    before = snap()
    payload = _build_multi_span_otlp_payload(span_count=5)
    resp = client.post(OTLP_TRACES_PATH, content=payload, headers=_PROTOBUF_HEADERS)
    after = snap()

    assert resp.status_code == 200
    assert delta(before, after, "spans") == 5
    assert delta(before, after, "errors") == 0
    assert delta(before, after, "duration_count") == 1
    assert delta(before, after, "payload_sum") > 0
    assert delta(before, after, "spans_per_req_sum") == 5

    # -- generic server error (RuntimeError) --
    _set_store(RuntimeError("DB connection lost"))
    before = snap()
    resp = client.post(OTLP_TRACES_PATH, content=_build_otlp_payload(), headers=_PROTOBUF_HEADERS)
    after = snap()

    assert resp.status_code == 422
    assert delta(before, after, "errors") == 1
    assert delta(before, after, "spans") == 0
    assert delta(before, after, "duration_count") == 1

    # -- NotImplementedError (501) --
    _set_store(NotImplementedError("not supported"))
    before = snap()
    resp = client.post(OTLP_TRACES_PATH, content=_build_otlp_payload(), headers=_PROTOBUF_HEADERS)
    after = snap()

    assert resp.status_code == 501
    assert delta(before, after, "errors") == 1
    assert delta(before, after, "duration_count") == 1

    # -- MlflowException 5xx increments error counter --
    _set_store(MlflowException("Persistence layer unavailable", error_code=INTERNAL_ERROR))
    before = snap()
    resp = client.post(OTLP_TRACES_PATH, content=_build_otlp_payload(), headers=_PROTOBUF_HEADERS)
    after = snap()

    assert resp.status_code == 500
    assert "Persistence layer unavailable" in resp.json()["message"]
    assert delta(before, after, "errors") == 1
    assert delta(before, after, "spans") == 0
    assert delta(before, after, "duration_count") == 1

    # -- MlflowException 4xx does NOT increment error counter --
    _set_store(MlflowException("Invalid parameter value", error_code=INVALID_PARAMETER_VALUE))
    before = snap()
    resp = client.post(OTLP_TRACES_PATH, content=_build_otlp_payload(), headers=_PROTOBUF_HEADERS)
    after = snap()

    assert resp.status_code == 400
    assert "Invalid parameter value" in resp.json()["message"]
    assert delta(before, after, "errors") == 0

    # -- client errors (bad content-type, malformed payload) never count as server errors --
    _set_store()
    before = snap()
    client.post(OTLP_TRACES_PATH, content=b"bad", headers=_PROTOBUF_HEADERS)
    client.post(
        OTLP_TRACES_PATH,
        content=b"x",
        headers={"Content-Type": "text/plain", "X-MLflow-Experiment-Id": "1"},
    )
    after = snap()

    assert delta(before, after, "errors") == 0
    assert delta(before, after, "duration_count") == 2

    # -- payload byte tracking --
    _set_store()
    payload = _build_otlp_payload()
    before = snap()
    client.post(OTLP_TRACES_PATH, content=payload, headers=_PROTOBUF_HEADERS)
    after = snap()

    assert delta(before, after, "payload_sum") == len(payload)

    # -- payload bytes tracks wire size even when compressed --
    payload = _build_multi_span_otlp_payload(span_count=50)
    compressed = gzip.compress(payload)
    before = snap()
    client.post(
        OTLP_TRACES_PATH,
        content=compressed,
        headers={**_PROTOBUF_HEADERS, "Content-Encoding": "gzip"},
    )
    after = snap()

    assert delta(before, after, "payload_sum") == len(compressed)

    # -- counters accumulate across requests --
    before = snap()
    client.post(
        OTLP_TRACES_PATH, content=_build_multi_span_otlp_payload(3), headers=_PROTOBUF_HEADERS
    )
    client.post(
        OTLP_TRACES_PATH, content=_build_multi_span_otlp_payload(7), headers=_PROTOBUF_HEADERS
    )
    after = snap()

    assert delta(before, after, "spans") == 10
    assert delta(before, after, "spans_per_req_sum") == 10


@pytest.mark.parametrize(
    ("content_type", "expected_media_type", "expected_body"),
    [
        ("application/x-protobuf", "application/x-protobuf", b""),
        ("application/json", "application/json", b"{}"),
        ("application/json; charset=utf-8", "application/json", b"{}"),
    ],
)
def test_otlp_response_encoding_matches_request(
    monkeypatch, content_type, expected_media_type, expected_body
):
    monkeypatch.setattr("mlflow.server.otel_api._get_tracking_store", lambda: _make_store()())
    if content_type.startswith("application/json"):
        span = {"traceId": "0" * 32, "spanId": "01" * 8, "name": "span"}
        content = json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]})
    else:
        content = _build_otlp_payload()

    resp = _make_test_client().post(
        OTLP_TRACES_PATH,
        content=content,
        headers={"Content-Type": content_type, "X-MLflow-Experiment-Id": "1"},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == expected_media_type
    assert resp.content == expected_body


def test_otlp_span_kind_preserved_in_batch_get(monkeypatch, tmp_path, db_uri):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    store = SqlAlchemyStore(db_uri, tmp_path.joinpath("artifacts").as_uri())
    monkeypatch.setattr("mlflow.server.handlers._tracking_store", store)

    trace_id = bytes.fromhex("0af7651916cd43dd8448eb211c80319c")
    request = ExportTraceServiceRequest()
    scope_spans = request.resource_spans.add().scope_spans.add()
    server_span = scope_spans.spans.add(
        trace_id=trace_id,
        span_id=bytes.fromhex("b7ad6b7169203331"),
        name="server-request",
        kind=OTelProtoSpan.SPAN_KIND_SERVER,
        start_time_unix_nano=1_000_000_000,
        end_time_unix_nano=4_000_000_000,
    )
    scope_spans.spans.add(
        trace_id=trace_id,
        span_id=bytes.fromhex("53995c3f42cd8ad8"),
        parent_span_id=server_span.span_id,
        name="outgoing-call",
        kind=OTelProtoSpan.SPAN_KIND_CLIENT,
        start_time_unix_nano=2_000_000_000,
        end_time_unix_nano=3_000_000_000,
    )

    # The full server app only accepts localhost Host headers by default.
    client = TestClient(create_fastapi_app(), base_url="http://localhost")
    response = client.post(
        OTLP_TRACES_PATH,
        content=request.SerializeToString(),
        headers={"Content-Type": "application/x-protobuf", "X-MLflow-Experiment-Id": "0"},
    )
    assert response.status_code == 200

    response = client.get(
        "/api/3.0/mlflow/traces/batchGet",
        params={"trace_ids": f"tr-{trace_id.hex()}"},
    )
    assert response.status_code == 200
    spans = response.json()["traces"][0]["spans"]
    assert {span["name"]: span.get("kind") for span in spans} == {
        "server-request": "SPAN_KIND_SERVER",
        "outgoing-call": "SPAN_KIND_CLIENT",
    }
