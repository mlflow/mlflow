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


# One probe per prefix-aware router; any non-404 response proves the route is registered.
_NATIVE_ROUTER_PROBES = (
    ("POST", OTLP_TRACES_PATH),
    ("POST", "/ajax-api/3.0/jobs/search"),
    ("POST", "/gateway/mlflow/v1/chat/completions"),
    ("GET", "/ajax-api/3.0/mlflow/assistant/config"),
)


def test_native_routers_registered_under_static_prefix(monkeypatch):
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/myprefix")
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    client = TestClient(create_fastapi_app())

    for method, path in _NATIVE_ROUTER_PROBES:
        assert client.request(method, f"/myprefix{path}", json={}).status_code != 404, path
        # The prefixed path replaces the unprefixed one, as for every Flask route.
        assert client.request(method, path, json={}).status_code == 404, path


def test_create_fastapi_app_rejects_template_static_prefix(monkeypatch):
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/{user}")
    with pytest.raises(MlflowException, match=r"must not contain"):
        create_fastapi_app()


def test_gateway_timing_header_present_for_prefixed_route(monkeypatch):
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/myprefix")
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    client = TestClient(create_fastapi_app())

    response = client.post("/myprefix/gateway/mlflow/v1/chat/completions", json={})
    assert MLFLOW_GATEWAY_DURATION_HEADER in response.headers
