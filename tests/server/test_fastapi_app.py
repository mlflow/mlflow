import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mlflow.server.fastapi_app import create_fastapi_app


@pytest.fixture
def client():
    return TestClient(create_fastapi_app())


def test_websocket_to_wsgi_mount_does_not_crash(client):
    # A WebSocket handshake routed to the catch-all WSGI mount must be rejected
    # cleanly (close code 1000) instead of raising AssertionError. Regression for #24146.
    with pytest.raises(WebSocketDisconnect) as excinfo:  # noqa: PT011
        with client.websocket_connect("/gateway/proxy/codex/v1/models"):
            pass
    assert excinfo.value.code == 1000


def test_http_to_wsgi_mount_still_served(client):
    resp = client.get("/health")
    assert resp.status_code == 200
