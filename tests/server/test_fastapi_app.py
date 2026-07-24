import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import MLFLOW_GATEWAY_DURATION_HEADER
from mlflow.server.fastapi_app import add_mcp_exception_handlers, create_fastapi_app
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH


@pytest.fixture
def client():
    return TestClient(create_fastapi_app())


def test_websocket_to_wsgi_mount_does_not_crash(client):
    # A WebSocket handshake routed to the catch-all WSGI mount must be rejected
    # cleanly (close code 1000) instead of raising AssertionError.
    with pytest.raises(WebSocketDisconnect) as excinfo:  # noqa: PT011
        with client.websocket_connect("/gateway/proxy/codex/v1/models"):
            pass
    assert excinfo.value.code == 1000


def test_http_to_wsgi_mount_still_served(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_mcp_exception_handler_delegates_for_non_mcp_routes():
    app = FastAPI()

    @app.exception_handler(MlflowException)
    async def existing_mlflow_exception_handler(request, exc):
        return JSONResponse(status_code=418, content={"detail": "delegated"})

    add_mcp_exception_handlers(app)

    @app.get("/non-mcp")
    async def non_mcp():
        raise MlflowException.invalid_parameter_value("boom")

    client = TestClient(app)
    response = client.get("/non-mcp")

    assert response.status_code == 418
    assert response.json() == {"detail": "delegated"}


# (method, path) pairs, one per mirrored router, that resolve without needing a
# valid body/auth — any non-404 response proves the route itself is registered.
_NATIVE_ROUTER_PROBES = (
    ("POST", OTLP_TRACES_PATH),
    ("POST", "/ajax-api/3.0/jobs/search"),
    ("POST", "/gateway/mlflow/v1/chat/completions"),
    ("GET", "/ajax-api/3.0/mlflow/assistant/config"),
)


def test_native_routers_not_prefixed_by_default(monkeypatch):
    monkeypatch.delenv(STATIC_PREFIX_ENV_VAR, raising=False)
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    client = TestClient(create_fastapi_app())

    for method, path in _NATIVE_ROUTER_PROBES:
        assert client.request(method, path, json={}).status_code != 404, path
        # No prefix is configured, so the prefixed mirror must not exist.
        assert client.request(method, f"/myprefix{path}", json={}).status_code == 404, path


def test_native_routers_also_registered_under_static_prefix(monkeypatch):
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/myprefix")
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    client = TestClient(create_fastapi_app())

    for method, path in _NATIVE_ROUTER_PROBES:
        # Unprefixed routes remain reachable directly (e.g. for local/direct access).
        assert client.request(method, path, json={}).status_code != 404, path
        # And are mirrored under the static prefix, matching how Flask routes and
        # `mcp_server_router` already behave.
        assert client.request(method, f"/myprefix{path}", json={}).status_code != 404, path


def test_static_prefix_trailing_slash_is_normalized(monkeypatch):
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/myprefix/")
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    client = TestClient(create_fastapi_app())

    response = client.post(f"/myprefix{OTLP_TRACES_PATH}", json={})
    assert response.status_code != 404
    # A doubled slash would mean the trailing slash on the prefix wasn't stripped.
    assert client.post(f"/myprefix/{OTLP_TRACES_PATH}", json={}).status_code == 404


def test_gateway_timing_header_present_for_prefixed_route(monkeypatch):
    # `add_gateway_timing_middleware` matches on the routed path; it must recognize
    # the prefixed mirror too, or prefixed gateway responses silently lose the
    # X-MLflow-Gateway-Duration-Ms header.
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/myprefix")
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    client = TestClient(create_fastapi_app())

    for path in (
        "/gateway/mlflow/v1/chat/completions",
        "/myprefix/gateway/mlflow/v1/chat/completions",
    ):
        response = client.post(path, json={})
        assert MLFLOW_GATEWAY_DURATION_HEADER in response.headers, path
